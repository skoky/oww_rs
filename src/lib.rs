use log::warn;
use log::info;
use crate::audio::AudioFeatures;
use crate::mic::MicHandler;
use crate::model::Model;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::thread::sleep;

pub mod audio;
mod tests;
pub mod model;
pub mod mic;

pub const CHUNK: usize = 1280;
pub const VOICE_SAMPLE_RATE: usize = 16000;
pub const BUFFER_SECS: usize = 4;

pub const RAW_BUFFER_SIZE: usize = VOICE_SAMPLE_RATE * BUFFER_SECS;

const CONVERSION_CONST: f32 = 32767.0;  // conversion between mic levels and model levels

const WAV_STORING_DIR: &str = "recordings";

const SLEEP_TIME_MS: usize = 50;


const CHUNK_RATE: u32 = 2;

const QUIET_THRESHOLD: f32 = 100.0;

pub async fn create_unlock_task(running: Arc<RwLock<bool>>, detection_threshold: f32) {
    let model_path = "ahoy_hugo.onnx";
    info!("Using unlock model {}", &model_path);

    while !*running.read().unwrap() == true {
        let model = Model::new(Path::new(model_path), detection_threshold);

        let audio = AudioFeatures::new();
        let mic = MicHandler::new(audio, model).unwrap();

        let r = running.clone();
        if let Err(e) = mic.loop_now(r) {
            warn!("Mic loop error {:?}. Reloading mic loop", e)
        }
        sleep(core::time::Duration::from_secs(1));
    }
}
