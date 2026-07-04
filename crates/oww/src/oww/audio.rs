use circular_buffer::CircularBuffer;
use log::trace;
use rten::{RunOptions, ThreadPool};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rust_embed::Embed;
use std::error::Error;
use std::sync::Arc;

use crate::ModelType;

#[derive(Embed)]
#[folder = "models/"]
struct Models;

pub const FEATURE_BUFFER_SIZE: usize = 16;
pub const MELS_BUFFER_SIZE: usize = 40;
const MEL_CIRC_SIZE: usize = 80 / 5;

#[derive(Debug)]
pub struct AudioFeaturesTract {
    mel: ModelType,
    emb: ModelType,
    pool: Arc<ThreadPool>,
    pub feature_buffer: Box<CircularBuffer<FEATURE_BUFFER_SIZE, Vec<f32>>>,
    pub mel_spectrogram_buffer: Box<CircularBuffer<MEL_CIRC_SIZE, Vec<f32>>>,
}

impl AudioFeaturesTract {
    pub fn new(pool: Arc<ThreadPool>) -> Self {
        let mel_model_data = Models::get("melspectrogram.onnx").unwrap().data.to_vec();
        let mel = Arc::new(rten::Model::load(mel_model_data).unwrap());

        let emb_model_data = Models::get("embedding_model.onnx").unwrap().data.to_vec();
        let emb = Arc::new(rten::Model::load(emb_model_data).unwrap());

        let mut feature_buffer = CircularBuffer::<FEATURE_BUFFER_SIZE, Vec<f32>>::boxed();
        for _ in 0..FEATURE_BUFFER_SIZE {
            feature_buffer.push_back(vec![0.0f32; 96]);
        }
        let mut mel_spectrogram_buffer = CircularBuffer::<MEL_CIRC_SIZE, Vec<f32>>::boxed();
        for _ in 0..MEL_CIRC_SIZE {
            mel_spectrogram_buffer.push_back(vec![0.0f32; 5 * 32]);
        }

        AudioFeaturesTract { mel, emb, pool, feature_buffer, mel_spectrogram_buffer }
    }

    pub fn create_default() -> Self {
        Self::new(Arc::new(ThreadPool::with_num_threads(1)))
    }

    fn run_opts(&self) -> RunOptions {
        RunOptions::default().with_thread_pool(Some(Arc::clone(&self.pool)))
    }

    pub fn get_melspectrogram(&mut self, data: &[f32]) -> Result<NdTensor<f32, 2>, Box<dyn Error>> {
        let input = NdTensor::from_data([1, 1280], data.to_vec());
        let input_id = self.mel.input_ids()[0];
        let output_id = self.mel.output_ids()[0];

        trace!("2:get_melspectrogram with data size {:?}", data.len());
        let [mel_out] = self.mel.run_n(
            vec![(input_id, input.view().into())],
            [output_id],
            Some(self.run_opts()),
        )?;

        let mel_tensor: NdTensor<f32, 4> = mel_out.try_into()?;
        trace!("2: get_melspectrogram with output tensor {:?}", mel_tensor.shape());
        let raw: Vec<f32> = mel_tensor.iter().copied().map(|v| (v / 10.0) + 2.0).collect();
        Ok(NdTensor::from_data([5, 32], raw))
    }

    pub fn get_audio_features(&mut self, data: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        trace!("2: data chunk: {:?}", data.len());
        let mel_chunk = self.get_melspectrogram(data)?;

        self.mel_spectrogram_buffer.push_back(mel_chunk.data().unwrap().to_vec());

        // Stack MEL_CIRC_SIZE frames of [5, 32] → flat [80*32] viewed as [80, 32]
        let stacked_data: Vec<f32> = self.mel_spectrogram_buffer
            .to_vec()
            .into_iter()
            .flatten()
            .collect();

        // Slice rows 4..80: skip first 4*32=128 elements → 76*32=2432 elements
        let smaller_data: Vec<f32> = stacked_data[4 * 32..].to_vec();

        // Run embedding model with input shape [1, 76, 32, 1]
        let emb_input = NdTensor::from_data([1, 76, 32, 1], smaller_data);
        let emb_input_id = self.emb.input_ids()[0];
        let emb_output_id = self.emb.output_ids()[0];

        let [emb_out] = self.emb.run_n(
            vec![(emb_input_id, emb_input.view().into())],
            [emb_output_id],
            Some(self.run_opts()),
        )?;

        // Output shape [1, 1, 1, 96] → extract 96 floats
        let emb_tensor: NdTensor<f32, 4> = emb_out.try_into()?;
        let emb_data: Vec<f32> = emb_tensor.iter().copied().collect();

        self.feature_buffer.push_back(emb_data);

        // Stack FEATURE_BUFFER_SIZE frames of 96 floats → flat [16*96=1536] viewed as [16, 96]
        let features: Vec<f32> = self.feature_buffer
            .to_vec()
            .into_iter()
            .flatten()
            .collect();

        Ok(features)
    }
}
