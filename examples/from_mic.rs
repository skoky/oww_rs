use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use log::info;
use oww_rs::audio::AudioFeatures;
use oww_rs::mic::MicHandler;
use oww_rs::model::Model;


const DETECTION_THRESHOLD: f32 = 0.3;  // 30%

fn main() {
    let running_c = Arc::new(RwLock::new(true));
    // ctrlc::set_handler(move || {
    //     println!("\nReceived Ctrl+C! Shutting down...");
    //     *running_c = false;
    // }).expect("Error setting Ctrl+C handler");

    let model_path = "hey_jarvis_v0.1.onnx";
    info!("Model {}", &model_path);
    let model = Model::new(Path::new(model_path), DETECTION_THRESHOLD);

    let audio = AudioFeatures::new();
    let mic = MicHandler::new(audio, model).unwrap();
    let locked_status = Arc::new(AtomicBool::new(true));

    mic.loop_now(running_c).unwrap();

    println!("Program terminated successfully");
}
