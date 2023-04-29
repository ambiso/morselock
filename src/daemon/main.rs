// Watches the sensor and compares it with the db

use std::{collections::VecDeque, error::Error, str::FromStr};

use morselock::{decode_morse_symbol_sequence, encode_letter, k_means_clustering, MorseSymbol};
use sqlx::{sqlite::SqliteConnectOptions, ConnectOptions};

trait Sensor {
    fn read(&mut self) -> Option<f64>;
}

struct SensorMock;

impl Sensor for SensorMock {
    fn read(&mut self) -> Option<f64> {
        Some(0.0)
    }
}

impl<T: Iterator<Item = f64>> Sensor for T {
    fn read(&mut self) -> Option<f64> {
        self.next()
    }
}

fn get_lengths(d: impl Iterator<Item = bool>, mode: bool) -> [i32; 2] {
    let mut run_lengths = Vec::new();
    let mut prev = false;
    let mut run_length = 1;
    for cur in d.skip_while(|x| *x == mode) {
        if cur == mode && prev == cur {
            run_length += 1;
        } else if prev != cur && prev == mode {
            run_lengths.push(run_length);
            run_length = 1;
        }
        prev = cur;
    }
    let mut lengths: [_; 2] = k_means_clustering(None, &run_lengths);
    lengths.sort();
    lengths
}

// fn get_lengths(run_lengths: , mode: bool) -> [i32; 2] {
//     let mut lengths: [_; 2] = k_means_clustering(&run_lengths);
//     lengths.sort();
//     lengths
// }

fn gaussian(x: f64, sigma: f64) -> f64 {
    let numerator = (-0.5 * x.powi(2)) / sigma.powi(2);
    let denominator = (2.0 * std::f64::consts::PI * sigma.powi(2)).sqrt();
    (numerator.exp()) / denominator
}

fn gaussian_kernel(kernel_size: usize, sigma: f64) -> Vec<f64> {
    let mut kernel = Vec::with_capacity(kernel_size);
    let half_kernel_size = (kernel_size / 2) as isize;

    for i in -(half_kernel_size as isize)..=(half_kernel_size as isize) {
        kernel.push(gaussian(i as f64, sigma));
    }

    kernel
}

fn gaussian_blur_average(image: &VecDeque<f64>, pixel_index: usize, kernel: &[f64]) -> f64 {
    let half_kernel_size = kernel.len() / 2;

    let mut blurred_pixel = 0.0;
    let mut kernel_sum = 0.0;

    for i in 0..kernel.len() {
        if let Some(index) = i
            .checked_sub(half_kernel_size)
            .and_then(|i| pixel_index.checked_add(i))
        {
            if index < image.len() {
                blurred_pixel += image[index] * kernel[i];
                kernel_sum += kernel[i];
            }
        }
    }

    blurred_pixel / kernel_sum
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let db_url = dotenvy::var("DATABASE_URL").unwrap_or("sqlite:db.sqlite".to_string());
    let mut db = SqliteConnectOptions::from_str(&db_url)?
        .create_if_missing(true)
        .connect()
        .await?;

    sqlx::migrate!().run(&mut db).await?;

    let raw_window_size = 200;
    let run_lengths_size = 20;
    let mut sensor_data = VecDeque::with_capacity(raw_window_size);
    let mut sum = 0.0;

    let mut input = vec![0.0; 50];
    {
        let dot_len = 10;
        let dash_len = 30;
        let inter_symbol_len = 10;
        let space_len = 20;
        input.extend("YOLO".chars().flat_map(|ch| {
            let encoded = encode_letter(ch).unwrap();
            let mut encoded: Vec<_> = encoded
                .iter()
                .flat_map(|symbol| match symbol {
                    MorseSymbol::Dot => {
                        let mut x = vec![1.0; dot_len];
                        x.extend(vec![0.0; inter_symbol_len]);
                        x
                    }
                    MorseSymbol::Dash => {
                        let mut x = vec![1.0; dash_len];
                        x.extend(vec![0.0; inter_symbol_len]);
                        x
                    }
                })
                .collect();
            encoded.extend(vec![0.0; space_len]);
            encoded
        }));
        input.extend(vec![0.0; 50]);
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0/4.0).unwrap();
        for x in input.iter_mut() {
            *x += normal.sample(&mut rand::thread_rng());
        }
    }

    let kernel_size = raw_window_size;
    let sigma = 7.0;
    let kernel = gaussian_kernel(kernel_size, sigma);

    use textplots::{Chart, Plot, Shape};

    println!("input");

    Chart::new(400, 32, 0.0, input.len() as f32)
        .lineplot(&Shape::Lines(
            input
                .iter()
                .enumerate()
                .map(|x| (x.0 as f32, *x.1 as f32))
                .collect::<Vec<_>>()
                .as_slice(),
        ))
        .display();
    let mut sensor = input.iter().cycle().copied();

    let mut symbols = Vec::with_capacity(10);
    let mut decoded = VecDeque::with_capacity(10);

    let mut dot_lengths = Vec::new();

    let mut run_length = 1;
    let mut run_lengths = VecDeque::with_capacity(run_lengths_size);

    let mut lengths: [Option<[i32; 2]>; 2] = [None, None];

    let mut n = 20;

    let mut blurred = Vec::new();

    loop {
        if sensor_data.len() >= raw_window_size {
            sum -= sensor_data.pop_front().unwrap();
            let mean = sum / sensor_data.len() as f64;
            let mid = sensor_data.len() / 2;
            let prev = gaussian_blur_average(&sensor_data, mid - 1, &kernel) > mean;
            let bl = gaussian_blur_average(&sensor_data, mid, &kernel);
            let new = bl > mean;
            blurred.push(bl);
            if new == prev {
                run_length += 1;
            } else {
                let max_space_length =
                    lengths[false as usize].map(|low_lengths| 4 * low_lengths[0]);
                if let Some(max_space_length) = max_space_length {
                    if !prev && run_length > max_space_length {
                        // discard space lengths longer than 5 times the dot length
                        eprintln!("Discarding run length {run_length} because it is too long");
                        run_length = max_space_length;
                    }
                }
                run_lengths.push_back((prev, run_length));
                run_length = 1;

                if run_lengths.len() >= run_lengths_size {
                    run_lengths.pop_front();

                    // update estimates of lengths
                    for mode in [false, true] {
                        let run_lengths = run_lengths
                            .iter()
                            .filter(|x| x.0 == mode)
                            .map(|x| x.1)
                            .collect::<Vec<_>>();

                        // dbg!(&run_lengths);
                        lengths[mode as usize] =
                            Some(k_means_clustering(lengths[mode as usize], &run_lengths));
                    }
                    if let [Some([inter_symbol_length, space_length]), Some([dot_length, dash_length])] =
                        lengths
                    {
                        dot_lengths.push(dot_length);
                        // dbg!(lengths);

                        let (symbol, run_length) = run_lengths[run_lengths.len() / 2];

                        eprintln!("Classifying {symbol} with run_length {run_length}");
                        // classify currently looked at run_length
                        if symbol {
                            // are we looking for dot/dash or for spacing?
                            if (run_length - dot_length).abs() < (run_length - dash_length).abs() {
                                // dot
                                eprintln!("Dot");
                                symbols.push(MorseSymbol::Dot);
                            } else {
                                // dash
                                eprintln!("Dash");
                                symbols.push(MorseSymbol::Dash);
                            }
                        } else {
                            if (run_length - inter_symbol_length).abs()
                                < (run_length - space_length).abs()
                            {
                                // inter symbol (awaiting next dot or dash)
                                eprintln!("Waiting for next symbol");
                            } else {
                                // finish this letter
                                // dbg!(&symbols);
                                n -= 1;
                                if n == 0 {
                                    break;
                                }
                                if let Some(letter) = decode_morse_symbol_sequence(&symbols) {
                                    decoded.push_back(letter);
                                }
                                if decoded.len() >= 5 {
                                    break;
                                }
                                symbols.clear();
                            }
                        }
                    }
                }
            }
        }
        use Sensor;
        if let Some(d) = sensor.read() {
            sum += d;
            sensor_data.push_back(d);
        } else {
            break;
        }
    }
    println!("blurred");

    Chart::new(400, 32, 0.0, blurred.len() as f32)
        .lineplot(&Shape::Lines(
            blurred
                .iter()
                .enumerate()
                .map(|x| (x.0 as f32, *x.1 as f32))
                .collect::<Vec<_>>()
                .as_slice(),
        ))
        .display();

    dbg!(decoded);

    Ok(())
}
