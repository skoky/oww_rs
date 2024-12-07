use crate::audio::AudioFeatures;
use crate::model::Model;
use crate::{BUFFER_SECS, CHUNK, DETECTIONS_PER_SEC, QUIET_THRESHOLD, VOICE_SAMPLE_RATE, WAV_STORING_DIR};
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

        let recorder = pv_recorder::PvRecorderBuilder::new(CHUNK as _)
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
        info!("Say the keyword used in model {:?}", &self.model.model_name);

        const RING_BUFFER_SIZE: usize = VOICE_SAMPLE_RATE * BUFFER_SECS;
        let mut _ring_buffer = CircularBuffer::<RING_BUFFER_SIZE, i16>::boxed();

        for _ in 0.._ring_buffer.capacity() {
            _ring_buffer.push_back(0);
        }

        self.recorder.start().expect("Failed to start audio recording");

        loop {
            let chunk: Vec<i16> = match self.recorder.read() {
                Ok(chunk) => chunk,
                Err(e) => {
                    warn!("Error reading chunk. Mic change? {}", e.to_string());
                    return Err(e.to_string());
                }
            };
            _ring_buffer.extend(&chunk);
            let continues_buffer = _ring_buffer.to_vec();

            let chunk_f32: Vec<f32> = chunk.to_vec().into_iter().map(|x| x as f32).collect();

            let rms = calculate_rms(&chunk_f32);
            if rms < QUIET_THRESHOLD {
                // TODO quite detected, sleep?
            }
            let (detected, prc) = self.model.detection(&mut self.audio, &chunk_f32)?;
            if detected {
                // TODO saving to wav file the continues_buffer if needed
            }
        }
    }
}


/// saves last 2 sec of ring buffer to a wav file and returns filename

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
    use crate::*;
    use circular_buffer::CircularBuffer;

    #[test]
    fn test_detection() {
        let mut audio = AudioFeatures::new();
        let mut model = Model::new(Path::new("hey_jarvis_v0.1.onnx"), 0.5);

        let mut ring_buffer = CircularBuffer::<64000, i16>::boxed();  // SAMPLE_RATE * 4 secs
        for _ in 0..64000 {  // prefill buffer as wav file is smaller than 4 secs
            ring_buffer.push_back(0);
        }

        let mut reader = hound::WavReader::open("hey_jarvis.wav").expect("Missing testing data");

        for (count, sample) in reader.samples::<i16>().enumerate() {  // have to convert from i16 to f32 for model
            ring_buffer.push_back(sample.unwrap());

            if count % CHUNK == 0 {
                let chunk = ring_buffer.range(ring_buffer.len() - CHUNK..).map(|v| *v as f32).collect();
                let (detected, percentage) = model.detection(&mut audio, &chunk).unwrap();
                if detected {
                    assert!(percentage > 0.8);
                    return;
                }
            }
        }
        assert!(false);
    }
}