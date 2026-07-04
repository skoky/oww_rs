use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, info, warn};
use oww_rs::config::SpeechUnlockType;
use audio_tools::mic_config::find_best_config;
use audio_tools::process_audio::resample_into_chunks;
use audio_tools::resampler::make_resampler;
use oww_rs::oww::{OwwModel, OWW_MODEL_CHUNK_SIZE};
use oww_rs::mic_cpal::build_input_stream;
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::thread;

/// Demonstrating the lib usage with cpal streaming from microphone.
///
/// Usage: cpal_test [alexa|ahoj_hugo|hey_mycroft]  (default: alexa)
fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::new().filter_level(log::LevelFilter::Info).init();

    let model_arg = std::env::args().nth(1).unwrap_or_else(|| "alexa".to_string());
    let (model_type, threshold, word) = match model_arg.as_str() {
        "ahoj_hugo"   => (SpeechUnlockType::OpenWakeWordAhojHugo,   0.3, "Ahoj Hugo"),
        "hey_mycroft" => (SpeechUnlockType::OpenWakeWordHeyMycroft,  0.5, "Hey Mycroft"),
        "alexa" | _   => (SpeechUnlockType::OpenWakeWordAlexa,       0.1, "Alexa"),
    };

    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    match device.description() {
        Ok(desc) => debug!("Input device: {}", desc.name()),
        Err(e) => { warn!("Couldn't get mic device: {:?}", e); exit(1); }
    }

    let (config, sample_format) = find_best_config(&device, false).unwrap();
    info!("Selected input config: {:?}", config);

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![]));
    let buffer_clone = buffer.clone();

    let original_sample_rate = config.sample_rate as f32;
    let channels = config.channels as usize;

    let err_fn = |err| warn!("An error occurred on the input stream: {}", err);

    let mut model = OwwModel::new(model_type, threshold).unwrap();
    let mut resampler = make_resampler(original_sample_rate as _, OWW_MODEL_CHUNK_SIZE as _, channels).unwrap();

    let (tx, rx) = mpsc::sync_channel(2);

    thread::spawn(move || {
        while let Ok(chunk) = rx.recv() {
            let d = model.detection(chunk);
            if d.detected {
                println!("Detected! {:?}", d);
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
                let _ = tx.try_send(chunk.data_f32.first().clone());
            }
        },
        err_fn,
        None,
    ).unwrap();

    stream.play()?;

    println!("Listening for `{word}`. Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
