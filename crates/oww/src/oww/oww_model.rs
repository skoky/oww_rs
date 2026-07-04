use crate::config::SpeechUnlockType;
use crate::model::Detection;
use crate::oww;
use crate::oww::OwwModel;
use crate::oww::audio::AudioFeaturesTract;
use circular_buffer::CircularBuffer;
use log::{debug, trace, warn};
use oww::DETECTION_BUFFER_SIZE;
use rten::{RunOptions, ThreadPool};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use rust_embed::Embed;
use std::{fs, io};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const MIN_POSITIVE_DETECTIONS: f32 = 3.0;
const NO_DETECTION_MS: u32 = 2_000;

#[derive(Embed)]
#[folder = "speech_models/"]
struct SpeechModels;

impl OwwModel {
    pub fn detection(&mut self, chunk_f32: Vec<f32>) -> Detection {
        let start = Instant::now();

        let audio_features = match self.audio.get_audio_features(chunk_f32.as_slice()) {
            Ok(features) => features,
            Err(e) => {
                warn!("Embeddings error {:?}", e);
                return crate::model::Detection {
                    detected: false,
                    probability: 0.0,
                    duration_ms: 0,
                };
            }
        };

        let (detected, prc) = self.detect(audio_features);

        let onnx_duration = start.elapsed();
        Detection {
            detected,
            probability: prc,
            duration_ms: onnx_duration.as_millis(),
        }
    }

    pub fn detect(&mut self, features: Vec<f32>) -> (bool, f32) {
        trace!("2: features size {:?}", features.len()); // 16*96 = 1536
        let input = NdTensor::from_data([1, 16, 96], features);
        trace!("2: inputs shape {:?}", input.shape()); // [1, 16, 96]

        let input_id = self.model.input_ids()[0];
        let output_id = self.model.output_ids()[0];
        let opts = RunOptions::default().with_thread_pool(Some(Arc::clone(&self.pool)));

        let [out] = self.model.run_n(
            vec![(input_id, input.view().into())],
            [output_id],
            Some(opts),
        ).unwrap();

        let out_tensor: NdTensor<f32, 2> = out.try_into().unwrap();
        trace!("2: output shape {:?}", out_tensor.shape()); // [1, 1]
        let probability = out_tensor.data().unwrap()[0];
        trace!("2: probability: {:?}", probability);

        self.detections_buffer.push_back(probability);

        let average_detection_probability = self.calculate_average();
        let since_last_detection = self.last_detection_time.elapsed().as_millis();

        // Detection fires on the falling edge — when probability drops below 0.1
        // while the running average of above-threshold frames is still high.
        if probability < 0.1 && average_detection_probability > self.threshold && since_last_detection > NO_DETECTION_MS as _ {
            self.last_detection_time = Instant::now();
            return (true, average_detection_probability);
        }
        if average_detection_probability > 0.1 {
            debug!("Prob {}, avg {} since {:?}", probability, average_detection_probability, since_last_detection);
        }
        (false, average_detection_probability)
    }

    fn calculate_average(&self) -> f32 {
        let all_detections = self.detections_buffer.to_vec();
        let mut detection_cumulative = 0.0;
        let mut positive_count = 0.0;
        for d in all_detections {
            if d > self.threshold {
                positive_count += 1.0;
                detection_cumulative += d;
            }
        }
        let avg = detection_cumulative / positive_count;
        if positive_count > MIN_POSITIVE_DETECTIONS && avg > self.threshold { avg } else { 0.0 }
    }

    pub fn new(model_type: SpeechUnlockType, threshold: f32) -> Result<OwwModel, String> {
        let model_data = match model_type {
            SpeechUnlockType::OpenWakeWordAlexa => {
                SpeechModels::get("alexa.onnx").unwrap().data.to_vec()
            }
            SpeechUnlockType::OpenWakeWordHeyMycroft => {
                SpeechModels::get("hey_mycroft_v0.1.onnx").unwrap().data.to_vec()
            }
            SpeechUnlockType::OpenWakeWordAhojHugo => {
                SpeechModels::get("ahoj_hugo.onnx").unwrap().data.to_vec()
            }
        };

        let model_unlock_word = match model_type {
            SpeechUnlockType::OpenWakeWordAlexa => "Alexa".to_string(),
            SpeechUnlockType::OpenWakeWordHeyMycroft => "Hey Mycroft".to_string(),
            SpeechUnlockType::OpenWakeWordAhojHugo => "Ahoj Hugo".to_string(),
        };
        let detections_buffer = CircularBuffer::<DETECTION_BUFFER_SIZE, f32>::new();
        let pool = Arc::new(ThreadPool::with_num_threads(1));

        let model = Arc::new(rten::Model::load(model_data).map_err(|e| e.to_string())?);
        Ok(OwwModel {
            audio: AudioFeaturesTract::new(Arc::clone(&pool)),
            model,
            pool,
            threshold,
            last_detection_time: Instant::now(),
            detections_buffer,
            model_unlock_word,
        })
    }

    pub fn from_file<P: AsRef<Path>>(path: P, model_unlock_word: String, threshold: f32) -> io::Result<OwwModel> {
        let model_data = fs::read(path)?;
        let detections_buffer = CircularBuffer::<DETECTION_BUFFER_SIZE, f32>::new();
        let pool = Arc::new(ThreadPool::with_num_threads(1));

        let model = Arc::new(
            rten::Model::load(model_data)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
        );
        Ok(OwwModel {
            audio: AudioFeaturesTract::new(Arc::clone(&pool)),
            model,
            pool,
            threshold,
            last_detection_time: Instant::now(),
            detections_buffer,
            model_unlock_word,
        })
    }
}
