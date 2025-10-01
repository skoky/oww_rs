mod tests {
    use crate::config::{SpeechUnlockType, UnlockConfig};
    use crate::model::{new_model};
    use crate::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
    use crate::{Models, load_wav};

    use circular_buffer::CircularBuffer;
    use log::{debug, info, trace, warn};

    
    use crate::mic::mic_cpal::handle_detect;
    use crate::mic::process_audio::{interlace_stereo, resample_into_chunks};
    use crate::mic::resampler::make_resampler;
    use crate::oww::audio::AudioFeaturesTract;
    
    use std::error::Error;
    
    use std::sync::{Arc, Mutex};
    
    use std::thread::sleep;
    use std::time::Duration;
    
    use tokio::sync::broadcast;
    use crate::chunk::ChunkType;
    use crate::config::SpeechUnlockType::OpenWakeWordAlexa;


    #[warn(dead_code)] // not really dead but used in tests
    fn test_detection(filename: &str) -> Result<bool, Box<dyn Error>> {
        // let mut audio = AudioFeatures::create_default();
        let mut audio2 = AudioFeaturesTract::create_default();
        let mut model = OwwModel::new(OpenWakeWordAlexa, 0.1)?;
        let mut chunk_buffer = CircularBuffer::<OWW_MODEL_CHUNK_SIZE, f32>::boxed();
        let _chunk_count = 0;
        for (chunk_count, s) in load_wav(filename).expect("Missing WAV file").into_iter().enumerate() {
            chunk_buffer.push_back(s);
            if chunk_count > 0 && chunk_count % OWW_MODEL_CHUNK_SIZE == 0 {
                sleep(Duration::from_millis(80)); // simulate data from mic
                let data = chunk_buffer.to_vec();
                // let features = audio.get_audio_features(data.clone())?;
                let features = audio2.get_audio_features(data.as_slice())?;
                let (detected2, prc2) = model.detect(features);
                // let (detected, prc) = model.detect(&features);
                // assert_eq!(detected, detected2);
                trace!("Detected: {:?}, prc {prc2}", detected2);
                if detected2 && prc2 > 0.5 {
                    info!("Detection {} -> {:?} chunk {}", filename, prc2, chunk_count);
                    return Ok(true);
                }
            }
        }
        info!("No detection found in {}", filename);
        Ok(false)
    }

    #[warn(dead_code)] // not really dead but used in tests
    fn test_detection_resampling3(filename: &str, original_sample_rate: usize, chunk_size: usize, channels: usize, model_type: SpeechUnlockType) -> Option<f32> {
        let buffer = Arc::new(Mutex::new(vec![]));

        let config = UnlockConfig {
            unlock_type: SpeechUnlockType::OpenWakeWordAlexa,
            detection_threshold: 0.5,
            quite_threshold: 10,
            endpoint_duration_secs: 0.5,
            save_wavs: false,
            unlock_time_secs: 30,
        };

        let model1 = match model_type {
            OpenWakeWordAlexa => new_model(config.clone()).unwrap(),
        };

        let model2 = match model_type {
            OpenWakeWordAlexa => new_model(config.clone()).unwrap(),
        };

        let model_fl = model1.frame_length();
        let models = Arc::new(Mutex::new(Models::new(model1, model2)));

        let mut resamplers = make_resampler(original_sample_rate as _, model_fl as _, channels).unwrap();
        trace!("Resampler {:?}", resamplers);

        let (chunks_sender, _) = broadcast::channel::<ChunkType>(1024);
        let quite_threshold = config.quite_threshold;
        let ring_buffer = Arc::new(Mutex::new(CircularBuffer::<10000000, f32>::boxed()));

        let mut wav = load_wav(filename).expect("Missing WAV file");
        for chunk in wav.chunks_mut(chunk_size) {
            let prb = resample_into_chunks(chunk, &buffer, channels, &mut resamplers)
                .iter()
                .map(|chunk2| {
                    if let Ok(mut ring_buffer) = ring_buffer.lock() {
                        for x in interlace_stereo(chunk2.clone().data_f32) {
                            ring_buffer.push_back(x);
                        }
                    }

                    let d = handle_detect(chunk2, chunks_sender.clone(), models.clone(), vec![], quite_threshold);
                    (d.probability * 100.0) as u32
                })
                .max();
            if prb.unwrap_or(0) > 0 {
                return prb.map(|v| v as f32 / 100.0);
            }
            // sleep(Duration::from_millis(80));
        }

        debug!("No detection in file {} and model {:?}", filename, model_type);
        None
    }
}
