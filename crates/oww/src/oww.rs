use crate::ModelType;
use crate::model::{Detection, Model};
use crate::oww::audio::AudioFeaturesTract;
use circular_buffer::CircularBuffer;
use rten::ThreadPool;
use std::sync::Arc;
use std::time::Instant;

pub mod audio;
mod oww_model;

pub const OWW_MODEL_CHUNK_SIZE: usize = 1280;

const DETECTION_BUFFER_SIZE: usize = 12; // 12 detection ~ 1 secs

#[derive(Debug)]
pub struct OwwModel {
    audio: AudioFeaturesTract,
    pub model: ModelType,
    pool: Arc<ThreadPool>,
    threshold: f32,
    pub last_detection_time: Instant,
    pub detections_buffer: CircularBuffer<DETECTION_BUFFER_SIZE, f32>,
    pub model_unlock_word: String,
}

impl Model for OwwModel {

    fn frame_length(&self) -> u32 {
        OWW_MODEL_CHUNK_SIZE as u32
    }

    fn detect(&mut self, chunk: Vec<f32>) -> Option<Detection> {
        Some(self.detection(chunk))
    }

    fn detect_i16(&mut self, _data: Vec<i16>) -> Option<Detection> {
        None
    }
}
