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
    let mut lengths: [_; 2] = k_means_clustering(&run_lengths);
    lengths.sort();
    lengths
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
    let thresholded_window_size = 400;
    let mut sensor_data = VecDeque::with_capacity(raw_window_size);
    let mut thresholded = VecDeque::with_capacity(thresholded_window_size);
    let mut sum = 0.0;

    let dot_len = 10;
    let dash_len = 30;
    let inter_symbol_len = 10;
    let space_len = 30;
    let mut input = vec![0.0; 50];
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

    use textplots::{Chart, Plot, Shape};

    input.extend(vec![0.0; 50]);

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


    // let mut prev = false;
    // let mut run_length = 0;
    // let mut run_lengths= VecDeque::with_capacity(thresholded_window_size);

    loop {
        if sensor_data.len() >= raw_window_size {
            sum -= sensor_data.pop_front().unwrap();
            let mean = sum / sensor_data.len() as f64;
            let new = sensor_data[sensor_data.len() / 2] > mean;
            thresholded.push_back(new);

            if thresholded.len() >= thresholded_window_size {
                thresholded.pop_front();
                // determine length of dots, dashes and spaces
                let [dot_length, dash_length] = get_lengths(thresholded.iter().copied(), true);
                let [inter_symbol_length, space_length] =
                    get_lengths(thresholded.iter().copied(), true);

                let prev = thresholded[thresholded.len() - 2];
                dot_lengths.push(space_length);
                if prev != new {
                    // decode symbol!
                    let run_length = thresholded
                        .iter()
                        .rev()
                        .skip(1)
                        .take_while(|x| **x == prev)
                        .count() as i32;
                    if prev {
                        if (run_length - dot_length).abs() < (run_length - dash_length).abs() {
                            // dot
                            symbols.push(MorseSymbol::Dot);
                        } else {
                            // dash
                            symbols.push(MorseSymbol::Dash);
                        }
                    } else {
                        if (run_length - inter_symbol_length).abs()
                            < (run_length - space_length).abs()
                        {
                            // inter symbol (awaiting next dot or dash)
                        } else {
                            // finish this letter
                            dbg!(&symbols);
                            if let Some(letter) = decode_morse_symbol_sequence(&symbols) {
                                decoded.push_back(letter);
                            }
                            if decoded.len() > 4 {
                                break;
                            }
                            symbols.clear();
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
    println!("dot_lengths");

    Chart::new(400, 32, 0.0, dot_lengths.len() as f32)
        .lineplot(&Shape::Lines(
            dot_lengths
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
