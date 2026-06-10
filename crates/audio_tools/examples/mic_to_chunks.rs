//! Microphone → 16 kHz chunk demo for the `audio_tools` crate.
//!
//! Opens the default input device, picks the best `cpal` config, then resamples
//! whatever the mic produces down to 16 kHz mono and slices it into fixed-size
//! `Chunk`s ready for a speech model. For each chunk it prints the running chunk
//! count and the chunk's RMS loudness.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p audio_tools --example mic_to_chunks
//! ```

use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use audio_tools::converters::i16_to_f32;
use audio_tools::mic_config::{default_input_device, find_best_config};
use audio_tools::process_audio::resample_into_chunks;
use audio_tools::resampler::make_resampler;
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, StreamTrait};

/// Fixed chunk size (in samples @ 16 kHz) a downstream model would expect.
/// 1280 samples == 80 ms, the cadence used by `oww-rs`.
const MODEL_CHUNK_SIZE: u32 = 1280;

fn main() -> Result<(), Box<dyn Error>> {
    // Find the default microphone and the best config for it.
    let device = default_input_device()?;
    let (config, sample_format) = find_best_config(&device, false)?;
    println!("Selected input config: {:?} ({:?})", config, sample_format);

    let channels = config.channels as usize;

    // Build the resampler chain: mic rate -> 16 kHz, in MODEL_CHUNK_SIZE slices.
    let mut resampler = make_resampler(config.sample_rate, MODEL_CHUNK_SIZE, channels)?;

    // Carry-over buffer for samples that don't yet fill a whole chunk.
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let chunk_count = Arc::new(AtomicU64::new(0));
    let chunk_count_cb = chunk_count.clone();

    let err_fn = |err| eprintln!("input stream error: {err}");

    // The processing closure is shared by both sample formats; the i16 branch
    // converts to f32 first so the rest of the pipeline is format-agnostic.
    let mut process = move |data: &[f32]| {
        let chunks = resample_into_chunks(data, &buffer, channels, &mut resampler);
        for chunk in chunks {
            let n = chunk_count_cb.fetch_add(1, Ordering::Relaxed) + 1;
            let samples = chunk.data_f32.first().len();
            println!("chunk #{n}: {samples} samples @ 16 kHz, rms {}", chunk.rms);
        }
    };

    let stream = match sample_format {
        SampleFormat::F32 => {
            device.build_input_stream(&config, move |data: &[f32], _| process(data), err_fn, None)?
        }
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let data_f32: Vec<f32> = data.iter().map(i16_to_f32).collect();
                process(&data_f32);
            },
            err_fn,
            None,
        )?,
        other => return Err(format!("unsupported sample format: {other:?}").into()),
    };

    stream.play()?;

    println!("Capturing and resampling to 16 kHz... Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
