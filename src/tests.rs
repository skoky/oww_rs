use log::info;
use crate::load_wav;
use crate::oww::audio::AudioFeaturesTract;
use crate::oww::OWW_MODEL_CHUNK_SIZE;

#[test]
fn test_mels2() {
    let mut audio = AudioFeaturesTract::create_default();

    let sample_data = [0f32; 1280];
    let mels = audio.get_melspectrogram(&sample_data).unwrap();
    assert_eq!(mels.shape(), [5, 32]);
}

