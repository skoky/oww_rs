use crate::audio::AudioFeatures;
use crate::mic::MicHandler;
use crate::model::Model;
use log::info;
use log::warn;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::thread::sleep;

pub mod audio;
mod tests;
pub mod model;
pub mod mic;

pub const CHUNK: usize = 1280;      // CHUNK size must be constant to be compatible with model inputs
pub const VOICE_SAMPLE_RATE: usize = 16000;   // the sample rate standard value for voice recording
pub const BUFFER_SECS: usize = 4;   // 4 secs buffering is enough for wake word, required by the model
pub const RAW_BUFFER_SIZE: usize = VOICE_SAMPLE_RATE * BUFFER_SECS;     // this is ring buffer size
const CONVERSION_CONST: f32 = 32767.0;  // conversion between mic levels, model levels and saving to wav file
const WAV_STORING_DIR: &str = "recordings";
const SLEEP_TIME_MS: usize = 50;  // sleep time between chunks. With 1280 chunks and 16kHz sampling rate we get chunk every 80ms
const CHUNK_RATE: u32 = 2;     // if value 2 is used, model detect every second chunk
const QUIET_THRESHOLD: f32 = 100.0;   // values under 100 means there is no soud from mic, no model trigering needed

pub fn create_unlock_task(model_path: &Path, detection_threshold: f32) -> Result<(), String> {
    info!("Using unlock model {}", &model_path.to_str().unwrap());

    let model = Model::new(&model_path, detection_threshold);

    let audio = AudioFeatures::new();
    let mic = MicHandler::new(audio, model).unwrap();

    mic.loop_now()
}
