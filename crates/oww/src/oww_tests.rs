
#[cfg(test)]
mod tests {
    use crate::config::SpeechUnlockType;
    use crate::load_wav;
    use crate::oww::audio::AudioFeaturesTract;
    use crate::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
    use audio_tools::process_audio::resample_into_chunks;
    use audio_tools::resampler::make_resampler;
    use rstest::rstest;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const DETECTION_THRESHOLD: f32 = 0.5;

    #[test]
    fn test_mels2() {
        let mut audio = AudioFeaturesTract::create_default();

        // one 1280-sample chunk + the 480-sample lookback = 8 mel frames
        let sample_data = [0f32; 1280];
        let mels = audio.get_melspectrogram(&sample_data).unwrap();
        assert_eq!(mels.shape(), [8, 32]);
    }

    /// Loads a fixture wav in whatever rate / channel layout it is stored in and
    /// returns it as mono (left channel) plus its sample rate.
    fn load_fixture_mono(wav_path: &str) -> (Vec<f32>, u32) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(wav_path);
        let spec = hound::WavReader::open(path).unwrap().spec();
        let interleaved = load_wav(wav_path).unwrap();
        let mono = match spec.channels {
            1 => interleaved,
            2 => interleaved.iter().step_by(2).copied().collect(),
            n => panic!("unsupported channel count {}", n),
        };
        (mono, spec.sample_rate)
    }

    /// Linear-interpolation resampler used only to derive the test input formats from
    /// the fixture wavs — deliberately independent of the rubato pipeline under test.
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

    /// Converts a fixture wav to the requested rate / channel layout, pads it with
    /// silence, resamples it back to 16 kHz through the audio_tools pipeline and feeds
    /// the 1280-sample chunks to the given wake-word model. Returns the number of
    /// detections.
    fn count_detections(wav_path: &str, model_type: SpeechUnlockType, sample_rate: u32, channels: usize) -> usize {
        let (fixture_mono, fixture_rate) = load_fixture_mono(wav_path);
        let mono = resample_linear(&fixture_mono, fixture_rate, sample_rate);
        let mut samples = match channels {
            1 => mono,
            2 => mono.iter().flat_map(|s| [*s, *s]).collect(),
            _ => panic!("unsupported channel count {}", channels),
        };

        // two seconds of trailing silence, so the probability can fall below 0.1 after
        // the wake word (detection fires on the falling edge) and the resampler buffers
        // are flushed
        let tail = vec![0f32; (sample_rate as usize * 2) * channels];
        samples.extend(tail);

        let mut model = OwwModel::new(model_type, DETECTION_THRESHOLD).unwrap();
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
    const HUGO_MAN_WAV: &str = "test_data/ahoj_hugo_man.wav";
    const HUGO_WOMAN_WAV: &str = "test_data/ahoj_hugo_woman.wav";

    use SpeechUnlockType::{OpenWakeWordAhojHugo, OpenWakeWordHeyJarvis};

    /// positive detections: every wake word must be detected at every rate / channel
    /// layout (3 clips × 3 rates × mono/stereo = 18 cases)
    #[rstest]
    #[case::jarvis(JARVIS_WAV, OpenWakeWordHeyJarvis)]
    #[case::hugo_man(HUGO_MAN_WAV, OpenWakeWordAhojHugo)]
    #[case::hugo_woman(HUGO_WOMAN_WAV, OpenWakeWordAhojHugo)]
    fn test_positive_detection(
        #[case] wav: &str,
        #[case] model_type: SpeechUnlockType,
        #[values(16000, 44100, 48000)] sample_rate: u32,
        #[values(1, 2)] channels: usize,
    ) {
        assert!(count_detections(wav, model_type, sample_rate, channels) >= 1);
    }

    /// negative detections: other speech must never trigger
    #[rstest]
    fn test_negative_detection(
        #[values(16000, 44100, 48000)] sample_rate: u32,
        #[values(1, 2)] channels: usize,
    ) {
        assert_eq!(count_detections(NEGATIVE_WAV, OpenWakeWordHeyJarvis, sample_rate, channels), 0);
    }

    /// Detection must not depend on where the wake word lands relative to the
    /// 1280-sample chunk boundaries: live mic audio arrives at arbitrary alignment.
    /// The mel front-end keeps a raw-audio lookback across chunks (mirroring
    /// openWakeWord's streaming melspectrogram), otherwise sub-chunk shifts of the
    /// same audio make the probability collapse.
    #[rstest]
    fn test_jarvis_alignment_offset(
        #[values(0, 160, 320, 480, 640, 800, 960, 1120)] offset: usize,
    ) {
        let mut samples = vec![0f32; offset];
        samples.extend(load_wav("hey_jarvis.wav").unwrap());
        samples.extend(vec![0f32; 16000]);

        let mut model = OwwModel::new(OpenWakeWordHeyJarvis, DETECTION_THRESHOLD).unwrap();
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
