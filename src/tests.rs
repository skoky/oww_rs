use crate::load_wav;
use crate::oww::OWW_MODEL_CHUNK_SIZE;
use crate::oww::audio::{AudioFeaturesTract, FEATURE_BUFFER_SIZE};
use log::info;
use ndarray::{Ix2, arr2};

#[test]
fn test_mels2() {
    let mut audio = AudioFeaturesTract::create_default();

    let sample_data = [0f32; 1280];
    let mels = audio.get_melspectrogram(&sample_data).unwrap();
    assert_eq!(mels.shape(), [5, 32]);
}

#[test]
fn test_embeddings2() {
    let mut audio = AudioFeaturesTract::create_default();

    let sample_data: Vec<f32> = load_wav("testing_data/testing_2x_hugo.wav").unwrap();

    info!("Sample data size {:?}", sample_data.len());
    let mut chunk_no = 1;
    for chunk in sample_data.chunks_exact(OWW_MODEL_CHUNK_SIZE) {
        let embeddings = audio.get_audio_features(chunk).unwrap();
        assert_eq!(embeddings.shape(), [16, 96]);
    }
}
