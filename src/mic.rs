use crate::audio::AudioFeatures;
use crate::model::Model;
use crate::{BUFFER_SECS, DETECTIONS_PER_SEC, QUIET_THRESHOLD, VOICE_SAMPLE_RATE, WAV_STORING_DIR};
use chrono::Utc;
use circular_buffer::CircularBuffer;
use log::{debug, info, warn};
use pv_recorder::PvRecorder;
use std::fs;

pub struct MicHandler {
    audio: AudioFeatures,
    model: Model,
    recorder: PvRecorder,
    save_recordings: bool,
}

impl MicHandler {
    pub fn new(audio: AudioFeatures, model: Model, save_recordings: bool) -> Result<Self, Box<dyn std::error::Error>> {
        // #[cfg(target_os = "windows")]
        // let lib_path = "winlib/libpv_recorder.dll";
        //
        // #[cfg(not(target_os = "windows"))]
        // let lib_path = "lib/libpv_recorder.dylib";  // to run on Mac in workspace

        let recorder = pv_recorder::PvRecorderBuilder::new((VOICE_SAMPLE_RATE as f32 / DETECTIONS_PER_SEC as f32) as i32)
            .buffered_frames_count(1)  // one frame is enough, we have local ring buffer
            // .library_path(Path::new(lib_path))
            .init().expect("PV recorder init error");
        info!("Mic device: {}", recorder.selected_device());

        Ok(MicHandler {
            audio,
            model,
            recorder,
            save_recordings,
        })
    }


    pub fn loop_now(&mut self) -> Result<(), String> {
        info!("Starting mic loop, listening, pv_recorder version {}", self.recorder.version());

        const RING_BUFFER_SIZE: usize = VOICE_SAMPLE_RATE * BUFFER_SECS;
        let mut ring_buffer = Box::new(CircularBuffer::<RING_BUFFER_SIZE, f32>::new());

        for _ in 0..ring_buffer.capacity() {
            ring_buffer.push_back(0.0);
        }

        self.recorder.start().expect("Failed to start audio recording");

        loop {
            let chunk = match self.recorder.read() {
                Ok(chunk) => chunk,
                Err(e) => {
                    warn!("Error reading chunk. Mic change? {}", e.to_string());
                    return Err(e.to_string());
                }
            };
            ring_buffer.extend_from_slice(chunk.iter().map(|v| *v as f32).collect::<Vec<f32>>().as_slice());

            let continues_buffer = ring_buffer.to_vec();
            let rms = calculate_rms(&continues_buffer);

            if rms > QUIET_THRESHOLD {
                self.detection(&continues_buffer)?;
            }
        }
    }

    fn detection(&mut self, continues_buffer: &Vec<f32>) -> Result<(bool, f32), String> {
        let embeddings = self.audio.get_embeddings(continues_buffer)?;
        let (detected, prc) = self.model.detect(&embeddings);
        if detected {
            let detection_prc = (prc * 100.0) as u32;
            info!("Detected {}%", detection_prc);
            if self.save_recordings {
                let _ = save_wav(continues_buffer, detection_prc as _);
            }
        }
        Ok((detected, prc))
    }
}


/// saves last 2 sec of ring buffer to a wav file and returns filename
fn save_wav(data: &[f32], detection_prc: u8) -> String {
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
    let two_secs = VOICE_SAMPLE_RATE * 2;
    let save_from_point = data.len() - two_secs;
    for (count, s) in (0_u32..).zip(data.iter()) {
        if count > save_from_point as _ {
            if let Err(e) = writer.write_sample(*s / i16::MAX as f32) {
                warn!("Error writing to wav file {:?}", e);
                break;
            }
        }
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
    (sum_of_squares / samples_tbd.len() as f64).sqrt() as f32
}

mod tests {
    use circular_buffer::CircularBuffer;
    use crate::*;

    #[test]
    fn test_detection() {
        let audio = AudioFeatures::new();
        let model = Model::new(Path::new("hey_jarvis_v0.1.onnx"), 0.5);
        let mut mic = MicHandler::new(audio, model, false).unwrap();
        let mut ring_buffer = CircularBuffer::<64000, f32>::boxed();  // SAMPLE_RATE * 4 secs
        for x in 0..64000 {  // prefill buffer as wav file is smaller than 4 secs
            ring_buffer.push_back(x as f32);
        }

        let mut reader = hound::WavReader::open("hey_jarvis.wav").expect("Missing testing data");
        for sample in reader.samples::<i16>() {  // have to convert from i16 to f32 for model
            ring_buffer.push_back(sample.unwrap() as f32 * i16::MAX as f32);
        }

        let (detected, percentage) = mic.detection(&ring_buffer.to_vec()).unwrap();
        assert!(detected);
        assert!(percentage > 0.8);
    }
}