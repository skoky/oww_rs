use rust_embed::Embed;
use crate::{BUFFER_SECS, VOICE_SAMPLE_RATE};
use ndarray::{s, Array, Array2, Array3, ArrayView, Axis, Dim, Ix2, IxDyn};
use ort::inputs;
use ort::Session;
use ort::SessionBuilder;

#[derive(Embed)]
#[folder = "models/"]
struct Models;

pub struct AudioFeatures {
    mel_session: Session,
    emb_session: Session,
}

impl AudioFeatures {
    pub fn new() -> Self {
        let mel_session = SessionBuilder::new().unwrap()
            // .with_profiling("prof.txt").unwrap()
            .commit_from_memory(&Models::get("melspectrogram.onnx").unwrap().data)
            .unwrap();
        let emb_session = SessionBuilder::new().unwrap()
            .commit_from_memory(&Models::get("embedding_model.onnx").unwrap().data).unwrap();

        AudioFeatures {
            mel_session,
            emb_session,
        }
    }

    pub fn get_melspectrogram(
        &mut self,
        data: &Array2<f32>,  // dim 1 = 64000; i.e. VOICE_SAMPLE_RATE * BUFFER_SECS
    ) -> Result<Array<f32, IxDyn>, String> {
        let input_tensor = inputs![data.clone()].unwrap();

        // Run model prediction
        let outputs = self.mel_session
            .run(input_tensor).unwrap();

        let tt = outputs["output"].try_extract_tensor::<f32>().unwrap();
        let ttt = tt.squeeze().mapv(|v| (v / 10.0) + 2.0);

        // dim is 397x32
        Ok(ttt)
    }

    pub(crate) fn get_embeddings(&mut self, data: &Vec<f32>) -> Result<Array2<f32>, String> {
        if data.len() != VOICE_SAMPLE_RATE * BUFFER_SECS {
            return Err("Invalid size".to_string());
        }
        let data_array = Array2::from_shape_vec((1, data.len()), data.iter().cloned().collect()).unwrap();
        let mels = self.get_melspectrogram(&data_array)?;  // dim 397x32
        let mel_x = mels.dim()[0];
        let window_size = 76;
        let step_size = 8;
        let mut windows = Array3::<f32>::zeros((0, 76, 32));

        for i in (0..mel_x).step_by(step_size) {
            if i + window_size <= mel_x {
                let window: ArrayView<f32, Dim<[usize; 2]>> = mels.slice(s![i..i + window_size, ..]);
                // let (_x, _y) = window.dim();
                let view3 = window.insert_axis(Axis(0));
                let x = windows.append(Axis(0), view3).unwrap();
            }
        }

        let windows2 = windows.insert_axis(Axis(3));
        let input_tensor = inputs! {"input_1"=> windows2}.unwrap();
        let outputs = self.emb_session.run(input_tensor).unwrap();
        let (_, v) = outputs.first_key_value().unwrap();
        let tt = v.try_extract_tensor::<f32>().unwrap();
        let ttt: Array<f32, IxDyn> = tt.squeeze().into_owned();
        let r = ttt.clone().into_dimensionality::<Ix2>().unwrap();
        Ok(r)
    }
}
