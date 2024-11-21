use crate::audio::AudioFeatures;
use crate::mic::MicHandler;
use crate::model::Model;
use log::info;
use std::path::Path;

pub mod audio;
mod tests;
pub mod model;
pub mod mic;


pub const VOICE_SAMPLE_RATE: usize = 16000;   // the sample rate standard value for voice recording
pub const BUFFER_SECS: usize = 4;   // 4 secs buffering is enough for wake word, required by the model

pub const DETECTIONS_PER_SEC: u32 = 4;   // more detections needs more CPU; less might be slower or miss wake word
const WAV_STORING_DIR: &str = "recordings";
const QUIET_THRESHOLD: f32 = 100.0;   // values under 100 means there is no soud from mic, no model trigering needed

pub fn create_unlock_task(model_path: &Path, detection_threshold: f32) -> Result<(), String> {
    info!("Using unlock model {}", &model_path.to_str().unwrap());

    let model = Model::new(&model_path, detection_threshold);

    let audio = AudioFeatures::new();
    let mut mic = MicHandler::new(audio, model, false).unwrap();

    mic.loop_now()
}
