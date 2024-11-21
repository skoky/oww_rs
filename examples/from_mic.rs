use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use log::info;
use oww_rs::audio::AudioFeatures;
use oww_rs::create_unlock_task;
use oww_rs::mic::MicHandler;
use oww_rs::model::Model;


const DETECTION_THRESHOLD: f32 = 0.3;  // 30%

fn main() {

    let model_path = Path::new("hey_jarvis_v0.1.onnx");

    create_unlock_task(&model_path, DETECTION_THRESHOLD).unwrap();

    println!("Program terminated successfully");
}
