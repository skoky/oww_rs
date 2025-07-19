use circular_buffer::CircularBuffer;
use ndarray::{s, stack, Array, Array1, Array2, Array4, Axis, Ix1, Ix2, IxDyn};
use ort::inputs;
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionOutputs};
use ort::value::{Tensor, TensorRef};
use rust_embed::Embed;
use crate::CHUNK;

#[derive(Embed)]
#[folder = "models/"]
struct Models;

const FEATURE_BUFFER_SIZE: usize = 41;
const MELS_BUFFER_SIZE: usize = 397; // ~1 sec

pub struct AudioFeatures {
    mel_session: Session,
    emb_session: Session,
    pub feature_buffer: Box<CircularBuffer<FEATURE_BUFFER_SIZE, Array<f32, Ix1>>>,
    pub mel_spectrogram_buffer: Box<CircularBuffer<397, Array1<f32>>>,
}

impl AudioFeatures {
    pub fn new() -> Self {
        let mel_session = SessionBuilder::new().unwrap()
            // .with_profiling("prof.txt").unwrap()
            .with_intra_threads(1).unwrap()
            .with_inter_threads(1).unwrap()
            .commit_from_memory(&Models::get("melspectrogram.onnx").unwrap().data)
            .unwrap();
        let emb_session = SessionBuilder::new().unwrap()
            .with_intra_threads(1).unwrap()
            .with_inter_threads(1).unwrap()
            .commit_from_memory(&Models::get("embedding_model.onnx").unwrap().data).unwrap();


        let mut feature_buffer = CircularBuffer::<FEATURE_BUFFER_SIZE, Array<f32, Ix1>>::boxed();
        for _ in 0..FEATURE_BUFFER_SIZE {
            feature_buffer.push_back(Array1::<f32>::zeros(96));
        }
        let mut mel_spectrogram_buffer: Box<CircularBuffer<MELS_BUFFER_SIZE, Array1<f32>>> =
            CircularBuffer::<MELS_BUFFER_SIZE, Array1<f32>>::boxed();
        for _ in 0..MELS_BUFFER_SIZE {
            mel_spectrogram_buffer.push_back(Array1::<f32>::zeros(32));
        }

        AudioFeatures {
            mel_session,
            emb_session,
            feature_buffer,
            mel_spectrogram_buffer,
        }
    }

    pub fn get_melspectrogram(
        &mut self,
        data: &Array2<f32>, // dim 1 = 64000; i.e. VOICE_SAMPLE_RATE * BUFFER_SECS
    ) -> Result<Array<f32, IxDyn>, String> {
        let input_tensor = inputs![TensorRef::from_array_view(data).unwrap()];

        // Run model prediction
        let outputs = self.mel_session.run(input_tensor).unwrap();

        let array1 = outputs["output"].try_extract_array::<f32>().unwrap();
        let array2 = array1.squeeze().mapv(|v| (v / 10.0) + 2.0);

        // dim is 5x32
        Ok(array2)
    }

    pub(crate) fn get_audio_features(&mut self, data: Vec<f32>) -> Result<Array2<f32>, String> {
        if data.len() != CHUNK {
            return Err(format!("Invalid raw size {}", data.len()));
        }

        let data_array = Array2::from_shape_vec((1, CHUNK), data).map_err(|e| e.to_string())?;

        let mels_chunk = self.get_melspectrogram(&data_array)?; // dim 5x32

        let mels = self.mel_convert_update_buffer(mels_chunk)?;

        let input = create_input_tensor(&mels);
        let input_tensor = inputs! {"input_1"=> Tensor::from_array(input).unwrap()};

        let outputs = self.emb_session.run(input_tensor).unwrap();

        let d = features_convert_and_update_buffer(&mut self.feature_buffer, outputs);
        d
    }

    fn mel_convert_update_buffer(&mut self, mels_chunk: Array<f32, IxDyn>) -> Result<Array<f32, Ix2>, String> {
        for row in mels_chunk.axis_iter(Axis(0)) {
            let x: Array<f32, Ix1> = row.into_dimensionality::<Ix1>().unwrap().to_owned();
            self.mel_spectrogram_buffer.push_back(x);
        }
        let arrays_ref: Vec<_> = self.mel_spectrogram_buffer.iter().map(|arr| arr.view()).collect();
        let mels = stack(Axis(0), &arrays_ref).map_err(|e| e.to_string())?;
        Ok(mels)
    }
}

fn features_convert_and_update_buffer(
    feature_buffer: &mut Box<CircularBuffer<41, Array<f32, Ix1>>>,
    outputs: SessionOutputs,
) -> Result<Array2<f32>, String> {
    for value in outputs.values() {
        let tt: ndarray::ArrayViewD<'_, f32> = value.try_extract_array::<f32>().unwrap();
        let ttt: Array<f32, Ix1> = tt.flatten().into_owned();
        feature_buffer.push_back(ttt);
        let vectors: Vec<Array<f32, Ix1>> = feature_buffer.to_vec();
        let a = ndarray::stack(Axis(0), &vectors.iter().map(|x| x.view()).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?;
        return Ok(a);
    }
    Err("No values from model".to_string())
}

fn create_input_tensor(mels: &Array<f32, Ix2>) -> Array4<f32> {
    let (mel_x, _) = mels.dim();
    let window_size = 76;
    let window = mels
        .slice(s![mel_x - window_size.., ..])
        .into_owned()
        .insert_axis(Axis(0))
        .insert_axis(Axis(3));
    window
}
