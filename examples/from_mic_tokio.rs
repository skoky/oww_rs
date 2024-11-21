use oww_rs::create_unlock_task;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use env_logger::Builder;
use log::{info, LevelFilter};

const DETECTION_THRESHOLD: f32 = 0.3;  // 30%

#[tokio::main]
async fn main() {

    Builder::new()
        .filter_level(LevelFilter::Debug)
        .init();

    let model_path = Path::new("hey_jarvis_v0.1.onnx");

    loop {
        let _ = create_unlock_task(&model_path, DETECTION_THRESHOLD);
        info!("Reloading PV recorder");
        sleep(Duration::from_secs(1));
    }
}
