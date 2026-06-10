use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, info, warn};
use oww_rs::config::SpeechUnlockType::OpenWakeWordHeyMycroft;
use audio_tools::mic_config::find_best_config;
use audio_tools::process_audio::resample_into_chunks;
use audio_tools::resampler::make_resampler;
use oww_rs::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
use std::process::exit;
use std::sync::{Arc, Mutex};

use oww_rs::mic_cpal::build_input_stream;

use std::sync::mpsc;
use std::thread;

/// Example using the Hey Mycroft model with cpal microphone input
fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    // Initialize CPAL
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    match device.description() {
        Ok(name) => {
            debug!("Input device: {}", name);
        }
        Err(e) => {
            warn!("Couldn't get mic device: {:?}", e);
            exit(1);
        }
    }

    let (mut config, sample_format) = find_best_config(&device, false).unwrap();
    // Prefer 48000 Hz for best real-time performance if available
    config.sample_rate = 48000;
    info!("Selected input config (forced 48kHz): {:?}", config);

    // Create a buffer to store audio data
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![]));
    let buffer_clone = buffer.clone();

    // Store the original sample rate and channels
    let original_sample_rate = config.sample_rate as f32;
    println!("{:?}", original_sample_rate);
    let channels = 1;

    // Create the input stream
    let err_fn = |err| warn!("An error occurred on the input stream: {}", err);

    let mut model = OwwModel::new(OpenWakeWordHeyMycroft, 0.5).unwrap();

    let mut resampler = make_resampler(
        original_sample_rate as _,
        OWW_MODEL_CHUNK_SIZE as _,
        channels,
    )
    .unwrap();

    let (tx, rx) = mpsc::sync_channel(100);

    thread::spawn(move || {
        while let Ok(chunk) = rx.recv() {
            let d = model.detection(chunk);
            if d.detected {
                println!("Result {:?}", d);
            } else {
                println!("Anything else1 {:?}", d);
            }
        }
    });

    let stream = build_input_stream(
        &device,
        &config,
        sample_format,
        move |data| {
            let chunks = resample_into_chunks(data, &buffer_clone, channels, &mut resampler);
            for chunk in chunks {
                if let Err(_) = tx.try_send(chunk.data_f32.first().clone()) {
                    warn!("Worker channel full, dropping chunk");
                }
            }
        },
        err_fn,
        None,
    ).unwrap();

    stream.play()?;

    println!("Recording and resampling to 16000 Hz... Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
