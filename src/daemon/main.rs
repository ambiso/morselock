// Watches the sensor and compares it with the db

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
use tokio::io::{AsyncReadExt};
use tokio::sync::mpsc as async_mpsc;

use std::{collections::VecDeque, error::Error, str::FromStr};

use futures::StreamExt;
use log::{debug, info, warn};
use morselock::{decode_morse_symbol_sequence, k_means_clustering, MorseSymbol};
use sqlx::{sqlite::SqliteConnectOptions, ConnectOptions};

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{InputCallbackInfo, StreamConfig};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;

use serde_derive::Deserialize;
use clap::{Parser, arg};

#[async_trait]
trait Sensor {
    async fn read(&mut self) -> Option<f64>;
}

struct SensorMock;

#[async_trait]
impl Sensor for SensorMock {
    async fn read(&mut self) -> Option<f64> {
        Some(0.0)
    }
}

#[async_trait]
impl<T: Iterator<Item = f64> + Send> Sensor for T {
    async fn read(&mut self) -> Option<f64> {
        self.next()
    }
}

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

async fn microphone_stream() -> impl Stream<Item = Vec<f32>> {
    let (async_tx, async_rx): (async_mpsc::Sender<Vec<f32>>, async_mpsc::Receiver<Vec<f32>>) =
        async_mpsc::channel(10);
    std::thread::spawn(|| {
        // Get the default host.
        let host = cpal::default_host();

        // Get the default input device (microphone).
        let input_device = host
            .default_input_device()
            .expect("Failed to get default input device");

        // Get the default input format.
        let input_format = input_device
            .default_input_config()
            .expect("Failed to get default input format");

        // Convert the input format to a stream configuration.
        let input_config: StreamConfig = input_format.into();

        // Create a channel to send the microphone data.
        // let (sync_tx, sync_rx): (mpsc::Sender<Vec<f32>>, mpsc::Receiver<Vec<f32>>) = mpsc::channel();
        // tokio::spawn(async move {
        //     forward_sync_to_async(sync_rx, async_tx).await;
        // });

        // Define a callback to process the microphone data.
        let callback = move |data: &[f32], _: &InputCallbackInfo| {
            let data = data.to_owned();
            let _ = async_tx.blocking_send(data);
        };

        // Create an input stream using the input device, stream configuration, and callback.
        let input_stream = input_device
            .build_input_stream(&input_config, callback, |_| {}, None)
            .expect("Failed to build input stream");

        // Play the input stream to start capturing microphone data.
        input_stream.play().expect("Failed to start input stream");

        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    // Wrap the Receiver in a Mutex and create a Tokio Stream.
    ReceiverStream::new(async_rx)
}
struct MicStream<T: Stream<Item = f32>> {
    inner: T,
}

#[async_trait]
impl<T: Stream<Item = f32> + Unpin + Send> Sensor for MicStream<T> {
    async fn read(&mut self) -> Option<f64> {
        self.inner.next().await.map(|x| x as f64)
    }
}

async fn microphone_sensor(sample_rate: usize) -> impl Sensor {
    let mic_stream = microphone_stream().await;

    MicStream {
        inner: mic_stream
            .flat_map(|x| futures::stream::iter(x))
            .chunks(44000 / sample_rate)
            .map(|x| x.iter().map(|x| x.abs()).sum::<f32>() / x.len() as f32), // downsample
    }
}


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opt {
    /// Number of raw samples to keep in memory. Used to predict whether new samples are a low or a high signal. Higher will take longer to adjust to changes, but it must be at least as long as a symbol (e.g. a dot-dash)
    #[arg(long, default_value = "500")]
    raw_window_size: usize,
    #[arg(long, default_value = "40")]
    /// The number of run-lengths to store (transitions between high and low). Used to estimate the symbol length. Higher memory will be more accurate, but will take longer to adjust to changes in the symbol length.
    run_length_memory_size: usize,
    #[arg(long, default_value = "3.0")]
    /// Sigma of the gaussian blur. Higher will make it more noise resistant, but can make it miss very short dit-s. 
    sigma: f64,
    #[arg(long, default_value = "50")]
    /// Number of samples to wait before processing it.
    lag: usize,
    #[arg(long, default_value = "0.9")]
    /// Discount factor on the threshold value. Closer to 1 will make changes in high/low thresholding slower.
    mean_discount_factor: f64,
    #[arg(long, default_value = "0.3")]
    /// Centered region to ignore high/low transitions in. (Between 0 and 1, values lower than 0.5 recommended)
    deadzone: f64,
    #[arg(long, default_value = "100")]
    /// Samples per second to downsample to
    sample_rate: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    console_subscriber::init();
    env_logger::init();
    let db_url = dotenvy::var("DATABASE_URL").unwrap_or("sqlite:db.sqlite".to_string());
    let mut db = SqliteConnectOptions::from_str(&db_url)?
        .create_if_missing(true)
        .connect()
        .await?;

    sqlx::migrate!().run(&mut db).await?;

    // let config_file = dotenvy::var("MORSELOCK_CONFIG").ok();
    // let opt = if let Some(fname) = config_file {
    //     let f = fs::File::open(fname).await?;
    //     let mut br = tokio::io::BufReader::new(f);
    //     let mut s = String::new();
    //     br.read_to_string(&mut s).await?;
    //     Opt::from_args_with_toml(&s)?
    // } else {
    //     Opt::parse()
    // };
    let opt = Opt::parse();

    // Play the input stream to start capturing microphone data.

    let mut sensor_data = VecDeque::with_capacity(opt.raw_window_size);

    // let mut input = vec![0.0; 50];
    // {
    //     let dot_len = 10;
    //     let dash_len = 30;
    //     let inter_symbol_len = 10;
    //     let space_len = 20;
    //     input.extend("YOLO".chars().flat_map(|ch| {
    //         let encoded = encode_letter(ch).unwrap();
    //         let mut encoded: Vec<_> = encoded
    //             .iter()
    //             .flat_map(|symbol| match symbol {
    //                 MorseSymbol::Dot => {
    //                     let mut x = vec![1.0; dot_len];
    //                     x.extend(vec![0.0; inter_symbol_len]);
    //                     x
    //                 }
    //                 MorseSymbol::Dash => {
    //                     let mut x = vec![1.0; dash_len];
    //                     x.extend(vec![0.0; inter_symbol_len]);
    //                     x
    //                 }
    //             })
    //             .collect();
    //         encoded.extend(vec![0.0; space_len]);
    //         encoded
    //     }));
    //     input.extend(vec![0.0; 50]);
    //     use rand_distr::{Distribution, Normal};
    //     let normal = Normal::new(0.0, 1.0 / 4.0).unwrap();
    //     for x in input.iter_mut() {
    //         *x += normal.sample(&mut rand::thread_rng());
    //     }
    // }

    let lag = opt.lag;
    let kernel_size = lag;
    let kernel = gaussian_kernel(kernel_size, opt.sigma);
    debug!("Kernel: {kernel:?}");

    // use textplots::{Chart, Plot, Shape};

    // let mut sensor = input.iter().cycle().copied();
    let mut sensor = microphone_sensor(opt.sample_rate).await;

    let mut symbols = Vec::with_capacity(10);
    let mut decoded = VecDeque::with_capacity(10);

    let mut dot_lengths = Vec::new();

    let mut run_length = 1;
    let mut run_lengths = VecDeque::with_capacity(opt.run_length_memory_size);

    let mut lengths: [Option<[i32; 2]>; 2] = [None, None];

    let mut blurred = Vec::new();
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        warn!("\nCtrl+C detected. Exiting...");
    })?;

    let mut prev = false;
    let mut discounted_mean = 0.0;
    let mut first = true;

    while running.load(Ordering::SeqCst) {
        use Sensor;
        if sensor_data.len() >= sensor_data.capacity() {
            sensor_data.pop_front();
        }
        if let Some(d) = sensor.read().await {
            sensor_data.push_back(d);
        } else {
            println!("No more data");
            break;
        }

        if sensor_data.len() < sensor_data.capacity() {
            // wait for warmup
            continue;
        }

        let mean = sensor_data.iter().sum::<f64>() / sensor_data.len() as f64;
        if first {
            discounted_mean = mean;
            first = false;
        } else {
            discounted_mean *= opt.mean_discount_factor;
            discounted_mean += (1.0 - opt.mean_discount_factor) * mean;
        }
        let min = sensor_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = sensor_data.iter().fold(-f64::INFINITY, |a, &b| a.max(b));
        let span = max - min;
        let deadzone = opt.deadzone; // dont make decisions in the middle 30%
        let deadzone_min = discounted_mean - span * deadzone / 2.;
        let deadzone_max = discounted_mean + span * deadzone / 2.;
        let mid = sensor_data.len() - lag;
        let bl = gaussian_blur_average(&sensor_data, mid, &kernel);
        let new = if bl > deadzone_min && bl < deadzone_max {
            // debug!("In deadzone with {deadzone_min} < {bl} < {deadzone_max} (data range: {min} - {max})");
            prev
        } else {
            bl > discounted_mean
        };
        blurred.push(new as i8);
        if new == prev {
            run_length += 1;
        } else {
            let low_dit_length = lengths[false as usize].map(|low_lengths| low_lengths[0]);
            if let Some(low_dit_length) = low_dit_length {
                let max_factor = 4;
                if !prev && run_length >= max_factor * low_dit_length && low_dit_length >= 1 {
                    // discard space lengths longer than max_factor times the dot length
                    debug!("Lowering long space: {run_length} long space; expected up to {max_factor} * {} = {}. Lowering to {}", low_dit_length, max_factor * low_dit_length, 3 * low_dit_length);
                    run_length = (3 * low_dit_length).min(opt.lag as i32);
                }
            }
            if run_lengths.len() >= run_lengths.capacity() {
                run_lengths.pop_front();
            }
            run_lengths.push_back((prev, run_length));
            run_length = 1;

            // update estimates of lengths
            for mode in [false, true] {
                let run_lengths = run_lengths
                    .iter()
                    .filter(|x| x.0 == mode)
                    .map(|x| x.1)
                    .collect::<Vec<_>>();

                if run_lengths.len() > 0 {
                    // dbg!(&run_lengths);
                    lengths[mode as usize] =
                        Some(k_means_clustering(lengths[mode as usize], &run_lengths));
                }
            }
            if let [Some([inter_symbol_length, space_length]), Some([dot_length, dash_length])] =
                lengths
            {
                let space_length = 3*dot_length;
                dot_lengths.push(dot_length);
                // dbg!(lengths);

                let (symbol, run_length) = run_lengths[run_lengths.len() - 1];

                debug!("Classifying {symbol} with run_length {run_length} using {lengths:?}");
                // classify currently looked at run_length
                if symbol {
                    // are we looking for dot/dash or for spacing?
                    if (run_length - dot_length).abs() < (run_length - dash_length).abs() {
                        // dot
                        debug!("Dot");
                        print!(".");
                        symbols.push(MorseSymbol::Dot);
                    } else {
                        // dash
                        debug!("Dash");
                        print!("-");
                        symbols.push(MorseSymbol::Dash);
                    }
                } else {
                    if (run_length - inter_symbol_length).abs() < (run_length - space_length).abs()
                    {
                        // inter symbol (awaiting next dot or dash)
                        // debug!("Waiting for next symbol");
                    } else {
                        // finish this letter
                        if let Some(letter) = decode_morse_symbol_sequence(&symbols) {
                            decoded.push_back(letter);
                            print!("{letter}");
                            let decoded_string: String = decoded.iter().collect();
                            info!("Decoded: {decoded_string}, discounted_mean: {discounted_mean}");
                        } else {
                            debug!("Invalid morse sequence");
                            print!("?");
                        }
                        if decoded.len() >= 10 {
                            decoded.pop_front();
                        }
                        symbols.clear();
                    }
                }
                std::io::stdout().flush().unwrap();
            }
        }
        prev = new;
    }

    use plotters::prelude::*;

    let root = BitMapBackend::new("blurred.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("blurred data", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-0f32..blurred.len() as f32, -0.3f32..1.3f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        blurred
            .iter()
            .enumerate()
            .map(|x| (x.0 as f32, *x.1 as f32)),
        &RED,
    ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
