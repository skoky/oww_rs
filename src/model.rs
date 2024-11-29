use ndarray::{Array2, Array3, Ix2};
use ort::{inputs, Session, SessionBuilder, SessionOutputs};
use std::path::Path;
use log::debug;

pub struct Model {
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
            session,
            threshold,
        }
    }
}


fn get_probability(out: SessionOutputs) -> f32 {
    let (_k, v) = out.first_key_value().unwrap();
    let t = v.try_extract_tensor::<f32>().unwrap();
    let tt = t.into_dimensionality::<Ix2>().unwrap();
    let prob = tt.as_slice().unwrap()[0];
    prob
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