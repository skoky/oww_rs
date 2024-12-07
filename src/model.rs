use crate::audio::AudioFeatures;
use log::{debug, info};
use ndarray::{Array2, Array3, Ix2};
use std::path::Path;
use log::warn;
use ort::inputs;
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionOutputs};

pub struct Model {
    pub model_name: String,
    pub session: Session,
    threshold: f32,
}

impl Model {
    pub(crate) fn detect(&self, embeddings: &Array2<f32>) -> (bool, f32) {
        let out = self.session.run(inputs![convert_array_shape(embeddings)].unwrap()).unwrap();
        let probability = get_probability(out);
        (probability > self.threshold, probability)
    }

    pub fn new(model_path: &Path, threshold: f32) -> Model {
        // let env = ort::init().commit().unwrap();
        let session = SessionBuilder::new().unwrap()
            // .with_profiling("prof.txt").unwrap()
            .with_intra_threads(1).unwrap()
            .with_inter_threads(1).unwrap()
            .commit_from_file(model_path)
            .unwrap();

        Model {
            model_name: model_path.to_str().unwrap().to_string(),
            session,
            threshold,
        }
    }


    pub(crate) fn detection(&mut self, mut audio: &mut AudioFeatures, continues_buffer: &Vec<f32>) -> Result<(bool, f32), String> {
        let embeddings = audio.get_audio_features(continues_buffer.clone())?;
        let (detected, prc) = self.detect(&embeddings);
        if detected {
            let detection_prc = (prc * 100.0) as u32;
            info!("Detected {}%", detection_prc);
        }
        Ok((detected, prc))
    }
}
fn get_probability(out: SessionOutputs) -> f32 {
    for v in out.values() {
        let array1 = v.try_extract_tensor::<f32>().unwrap();
        let array2 = array1.into_dimensionality::<Ix2>().unwrap();
        return match array2.as_slice() {
            None =>  {
                warn!("No prob from model");
                0.0
            }
            Some(p) if p.len() == 1 => {
                p[0]
            }
            _ => {
                warn!("Invalid output from model");
                0.0
            }
        };
    }
    warn!("No values probability found from model");
    0.0
}


fn convert_array_shape(input: &Array2<f32>) -> Array3<f32> {
    assert_eq!(input.shape(), &[41, 96], "Input array must be shape (41,96)");

    // Create output array
    let mut output = Array3::<f32>::zeros((1, 16, 96));

    // Copy data from input to output
    // Note: This will only copy the last 16 rows of data
    let offset = 41 - 16;
    for i in offset..41 {
        for j in 0..96 {
            output[[0, i - (offset), j]] = input[[i, j]];
        }
    }
    output
}