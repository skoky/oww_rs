use crate::{BUFFER_SECS, CHUNK, CHUNK_RATE, CONVERSION_CONST, QUIET_THRESHOLD, SLEEP_TIME_MS, VOICE_SAMPLE_RATE, WAV_STORING_DIR};
use crate::audio::AudioFeatures;
use crate::model::Model;
use chrono::Utc;
use circular_buffer::CircularBuffer;
use log::{debug, error, info, warn};
use pv_recorder::PvRecorder;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::sleep;
use std::time::{Duration, Instant};
use ndarray::ErrorKind::IncompatibleShape;

pub struct MicHandler {
    pub audio: AudioFeatures,
    pub model: Model,
    pub recorder: PvRecorder,
}

impl MicHandler {
    pub fn new(audio: AudioFeatures, model: Model) -> Result<Self, Box<dyn std::error::Error>> {
        // #[cfg(target_os = "windows")]
        // let lib_path = "winlib/libpv_recorder.dll";
        //
        // #[cfg(not(target_os = "windows"))]
        // let lib_path = "lib/libpv_recorder.dylib";  // to run on Mac in workspace

        // TODO test increasing frame_length to CHUNK * X to improve performance
        let recorder = pv_recorder::PvRecorderBuilder::new(CHUNK as _)
            // .library_path(Path::new(lib_path))
            .init().expect("PV recorder init error");
        info!("Mic device: {}", recorder.selected_device());

        Ok(MicHandler {
            audio,
            model,
            recorder,
        })
    }

    pub fn loop_now(mut self, running: Arc<RwLock<bool>>) -> Result<(), String> {
        println!("Starting mic loop, listening, version {}", self.recorder.version());
        let mut ring_buffer = Box::new(CircularBuffer::<64000, f32>::new());  // FIXME
        for _ in 0..ring_buffer.capacity() {
            ring_buffer.push_back(0.0);
        }

        self.recorder.start().expect("Failed to start audio recording");
        let mut chunk_count: u32 = 0;

        while *running.read().unwrap() {
            let start = Instant::now();
            let chunk = self.recorder.read().expect("Error reading chunk");
            ring_buffer.extend_from_slice(chunk.iter().map(|v| *v as f32).collect::<Vec<f32>>().as_slice());
            if chunk_count % CHUNK_RATE == 0 {
                let continues_buffer = ring_buffer.to_vec();
                if calculate_rms(&continues_buffer) > QUIET_THRESHOLD {
                    let embeddings = self.audio.get_embeddings(&continues_buffer)?;
                    let (detected, prc) = self.model.detect(&embeddings);
                    if detected {
                        let detection_prc = (prc * 100.0) as u32;
                        println!("Detected {}%", detection_prc);
                        let _ = save_wav(&continues_buffer, detection_prc as _);
                    }
                }
            }
            chunk_count += 1;
            // println!("Chunk {} Loop duration {:?}", chunk_count, start.elapsed());
            sleep(Duration::from_millis(SLEEP_TIME_MS as u64)); // TODO better version
        }
        Ok(())
    }
}


/// saves last 2 sec of ring buffer to a wav file and returns filename
fn save_wav(data: &Vec<f32>, detection_prc: u8) -> String {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: VOICE_SAMPLE_RATE as _,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let now = Utc::now();
    let ts = now.format("%y%m%d-%H%M%S").to_string();

    fs::create_dir_all(WAV_STORING_DIR).unwrap();
    let filename = format!("{WAV_STORING_DIR}/recording_{}_{}.wav", detection_prc, ts);
    let mut writer = hound::WavWriter::create(&filename, spec).unwrap();
    let mut count: u32 = 0;
    let two_secs = VOICE_SAMPLE_RATE * 2;
    let save_from_point = data.len() - two_secs;
    for s in data.iter() {
        if count > save_from_point as _ {
            if let Err(e) = writer.write_sample(*s / CONVERSION_CONST) {
                warn!("Error writing to wav file {:?}", e);
                break;
            }
        }
        count += 1;
    }
    debug!("Recording saved to {filename}");
    filename
}

/// calculate RMS from 1 sec of data
fn calculate_rms(samples: &Vec<f32>) -> f32 {
    let samples_tbd = if samples.len() > VOICE_SAMPLE_RATE {
        let len_tbd = VOICE_SAMPLE_RATE * (BUFFER_SECS - 1);
        &samples[len_tbd..]
    } else {
        &samples
    };
    let sum_of_squares: f64 = samples_tbd.iter().map(|&s| s as f64 * s as f64).sum();
    let rms = (sum_of_squares / samples_tbd.len() as f64).sqrt() as f32;
    // println!("Rms {}", rms);
    rms
}
