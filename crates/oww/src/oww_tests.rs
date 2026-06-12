
#[cfg(test)]
mod tests {
    use crate::config::SpeechUnlockType;
    use crate::load_wav;
    use crate::oww::audio::AudioFeaturesTract;
    use crate::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
    use audio_tools::process_audio::resample_into_chunks;
    use audio_tools::resampler::make_resampler;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const DETECTION_THRESHOLD: f32 = 0.5;
    const FIXTURE_SAMPLE_RATE: u32 = 48000;

    #[test]
    fn test_mels2() {
        let mut audio = AudioFeaturesTract::create_default();

        // one 1280-sample chunk + the 480-sample lookback = 8 mel frames
        let sample_data = [0f32; 1280];
        let mels = audio.get_melspectrogram(&sample_data).unwrap();
        assert_eq!(mels.shape(), [8, 32]);
    }

    /// Loads a 48 kHz stereo fixture wav and returns its left channel as 48 kHz mono.
    fn load_fixture_mono(wav_path: &str) -> Vec<f32> {
        let interleaved = load_wav(wav_path).unwrap();
        interleaved.iter().step_by(2).copied().collect()
    }

    /// Linear-interpolation resampler used only to derive the test input formats from
    /// the 48 kHz fixture — deliberately independent of the rubato pipeline under test.
    fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if src_rate == dst_rate {
            return samples.to_vec();
        }
        let n_out = (samples.len() as u64 * dst_rate as u64 / src_rate as u64) as usize;
        (0..n_out)
            .map(|i| {
                let pos = i as f64 * src_rate as f64 / dst_rate as f64;
                let i0 = pos as usize;
                let i1 = (i0 + 1).min(samples.len() - 1);
                let frac = (pos - i0 as f64) as f32;
                samples[i0] * (1.0 - frac) + samples[i1] * frac
            })
            .collect()
    }

    /// Converts the 48 kHz stereo fixture to the requested rate / channel layout, pads
    /// it with silence, resamples it back to 16 kHz through the audio_tools pipeline
    /// and feeds the 1280-sample chunks to the "hey jarvis" model. Returns the number
    /// of detections.
    fn count_detections(wav_path: &str, sample_rate: u32, channels: usize) -> usize {
        let mono = resample_linear(&load_fixture_mono(wav_path), FIXTURE_SAMPLE_RATE, sample_rate);
        let mut samples = match channels {
            1 => mono,
            2 => mono.iter().flat_map(|s| [*s, *s]).collect(),
            _ => panic!("unsupported channel count {}", channels),
        };

        // two seconds of trailing silence, so the probability can fall below 0.1 after
        // the wake word (detection fires on the falling edge) and the resampler buffers
        // are flushed. No leading padding: the mel front-end processes each 1280-sample
        // chunk independently, so detection is sensitive to sub-chunk alignment shifts
        let tail = vec![0f32; (sample_rate as usize * 2) * channels];
        samples.extend(tail);

        let mut model = OwwModel::new(SpeechUnlockType::OpenWakeWordHeyJarvis, DETECTION_THRESHOLD).unwrap();
        // the refractory gate (NO_DETECTION_MS) measures wall time since model creation;
        // rewind it so a detection within the clip is not suppressed
        model.last_detection_time = Instant::now() - Duration::from_secs(10);

        let mut resamplers = make_resampler(sample_rate, OWW_MODEL_CHUNK_SIZE as u32, channels).unwrap();
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let chunks = resample_into_chunks(&samples, &buffer, channels, &mut resamplers);

        let mut detections = 0;
        for chunk in chunks {
            let mono = chunk.data_f32[0].clone();
            assert_eq!(mono.len(), OWW_MODEL_CHUNK_SIZE);
            if model.detection(mono).detected {
                detections += 1;
            }
        }
        detections
    }

    const JARVIS_WAV: &str = "test_data/hey_jarvis_48k_stereo.wav";
    const NEGATIVE_WAV: &str = "test_data/negative_48k_stereo.wav";

    // positive detections: "hey jarvis" must be detected at every rate / channel layout

    #[test]
    fn test_jarvis_16k_mono() {
        assert!(count_detections(JARVIS_WAV, 16000, 1) >= 1);
    }

    #[test]
    fn test_jarvis_16k_stereo() {
        assert!(count_detections(JARVIS_WAV, 16000, 2) >= 1);
    }

    #[test]
    fn test_jarvis_44_1k_mono() {
        assert!(count_detections(JARVIS_WAV, 44100, 1) >= 1);
    }

    #[test]
    fn test_jarvis_44_1k_stereo() {
        assert!(count_detections(JARVIS_WAV, 44100, 2) >= 1);
    }

    #[test]
    fn test_jarvis_48k_mono() {
        assert!(count_detections(JARVIS_WAV, 48000, 1) >= 1);
    }

    #[test]
    fn test_jarvis_48k_stereo() {
        assert!(count_detections(JARVIS_WAV, 48000, 2) >= 1);
    }

    /// Detection must not depend on where the wake word lands relative to the
    /// 1280-sample chunk boundaries: live mic audio arrives at arbitrary alignment.
    /// The mel front-end keeps a raw-audio lookback across chunks (mirroring
    /// openWakeWord's streaming melspectrogram), otherwise sub-chunk shifts of the
    /// same audio make the probability collapse.
    #[test]
    fn test_jarvis_alignment_offsets() {
        let jarvis = load_wav("hey_jarvis.wav").unwrap();
        for offset in (0..1280).step_by(160) {
            let mut samples = vec![0f32; offset];
            samples.extend(&jarvis);
            samples.extend(vec![0f32; 16000]);

            let mut model = OwwModel::new(SpeechUnlockType::OpenWakeWordHeyJarvis, DETECTION_THRESHOLD).unwrap();
            let mut max_prob = 0f32;
            for chunk in samples.chunks_exact(OWW_MODEL_CHUNK_SIZE) {
                let d = model.detection(chunk.to_vec());
                if d.probability > max_prob {
                    max_prob = d.probability;
                }
            }
            assert!(
                max_prob > DETECTION_THRESHOLD,
                "wake word missed at offset {} samples: max avg probability {:.4}",
                offset,
                max_prob
            );
        }
    }

    // negative detections: other speech must never trigger

    #[test]
    fn test_negative_16k_mono() {
        assert_eq!(count_detections(NEGATIVE_WAV, 16000, 1), 0);
    }

    #[test]
    fn test_negative_16k_stereo() {
        assert_eq!(count_detections(NEGATIVE_WAV, 16000, 2), 0);
    }

    #[test]
    fn test_negative_44_1k_mono() {
        assert_eq!(count_detections(NEGATIVE_WAV, 44100, 1), 0);
    }

    #[test]
    fn test_negative_44_1k_stereo() {
        assert_eq!(count_detections(NEGATIVE_WAV, 44100, 2), 0);
    }

    #[test]
    fn test_negative_48k_mono() {
        assert_eq!(count_detections(NEGATIVE_WAV, 48000, 1), 0);
    }

    #[test]
    fn test_negative_48k_stereo() {
        assert_eq!(count_detections(NEGATIVE_WAV, 48000, 2), 0);
    }
}
