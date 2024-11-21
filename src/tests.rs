mod tests {
    use crate::*;
    use ndarray::{arr2, Ix2};

    #[test]
    fn test_mels() {
        let mut audio = AudioFeatures::new();

        let sample_data = [0f32; 64000];
        let mels = audio.get_melspectrogram(&arr2(&[sample_data])).unwrap();
        let dim = mels.clone().into_dimensionality::<Ix2>().unwrap().dim();
        assert_eq!(dim, (397, 32));
    }

    #[test]
    fn test_embeddings() {
        let mut audio = AudioFeatures::new();
        let sample_data = [0f32; VOICE_SAMPLE_RATE * BUFFER_SECS];
        let embeddings = audio.get_embeddings(&sample_data.to_vec()).unwrap();
        let dim = embeddings.into_dimensionality::<Ix2>().unwrap().dim();
        assert_eq!(dim, (41, 96))
    }
}
