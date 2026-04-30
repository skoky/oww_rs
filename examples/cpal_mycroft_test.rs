use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, info, warn};
use oww_rs::config::SpeechUnlockType::OpenWakeWordHeyMycroft;
use oww_rs::mic::converters::i16_to_f32;
use oww_rs::mic::mic_config::find_best_config;
use oww_rs::mic::process_audio::resample_into_chunks;
use oww_rs::mic::resampler::make_resampler;
use oww_rs::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
use std::process::exit;
use std::sync::{Arc, Mutex};

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

    let (mut config, sample_format) = find_best_config(&device).unwrap();
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

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                let chunks = resample_into_chunks(data, &buffer_clone, channels, &mut resampler);
                for chunk in chunks {
                    let d = model.detection(chunk.data_f32.first().clone());
                    if d.detected {
                        println!("Result f32 {:?}", d);
                    } else {
                        println!("Anything else1 {:?}", d);
                    }
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &_| {
                // Convert i16 to f32
                let samples: Vec<f32> = data.iter().map(i16_to_f32).collect();
                let chunks =
                    resample_into_chunks(&samples, &buffer_clone, channels, &mut resampler);
                for chunk in chunks {
                    let d = model.detection(chunk.data_f32.first().clone());
                    if d.detected {
                        println!("Result i16 {:?}", d);
                    } else {
                        println!("Anything else2");
                    }
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &config,
            move |_data: &[u16], _: &_| {
                panic!("U16 format is not supported");
            },
            err_fn,
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;

    println!("Recording and resampling to 16000 Hz... Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
