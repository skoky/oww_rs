mod tests {
    use crate::config::{SpeechUnlockType, UnlockConfig};
    use crate::mic::converters::{f32_to_i16, i16_to_f32};
    use crate::model::{Model, new_model};
    use crate::oww::{OWW_MODEL_CHUNK_SIZE, OwwModel};
    use crate::{Models, VOICE_SAMPLE_RATE, load_wav};

    use circular_buffer::CircularBuffer;
    use log::{debug, info, trace, warn};

    use crate::mic::RING_BUFFER_SIZE;
    use crate::mic::mic_cpal::{MODEL_SAMPLE_RATE, handle_detect};
    use crate::mic::process_audio::{interlace_stereo, resample_into_chunks};
    use crate::mic::resampler::{make_resampler, resample_audio};
    use crate::oww::audio::AudioFeaturesTract;
    use crate::save::{save_full_wav, save_full_wav_with_channels};
    use std::error::Error;
    use std::sync::mpsc::channel;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::thread::sleep;
    use std::time::Duration;
    use approx::assert_relative_eq;
    use tokio::sync::broadcast;
    use crate::chunk::ChunkType;
    use crate::config::SpeechUnlockType::OpenWakeWordAlexa;

    #[test]
    fn test_detection10() {
        assert!(test_detection("testing_data/testing_2x_hugo.wav").unwrap());
    }
    #[test]
    fn test_detection11() {
        assert!(!test_detection("testing_data/ahoj_ufo.wav").unwrap());
    }

    #[test]
    fn test_detection12() {
        assert!(!test_detection("testing_data/jake_je_pocasi.wav").unwrap());
    }

    #[test]
    fn test_detection13() {
        assert!(!test_detection("testing_data/short_silence.wav").unwrap());
    }

    #[test]
    fn test_detection1() -> Result<(), Box<dyn std::error::Error>> {
        // with music!
        assert!(test_detection("testing_data/p_rhino_surface1_100_250123-084742_1293.wav")?);
        Ok(())
    }
    #[test]
    fn test_detection2() {
        assert!(test_detection("testing_data/p_rhino_surface2_100_241227-164451_short.wav").unwrap());
    }

    #[test]
    fn test_detection3() {
        assert!(test_detection("testing_data/p_rhino_surface2_100_241227-164451_short2.wav").unwrap());
        // 79% only!
    }

    #[test]
    fn test_detection4() {
        assert!(test_detection("testing_data/p_rhino_surface2_100_241227-164451.wav").unwrap());
    }

    #[test]
    fn test_detection5() {
        assert!(test_detection("testing_data/p_rhino_surface2_100_241227-164451_hugo_2x.wav").unwrap());
    }

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

    #[test]
    fn test_detection_with_resampling() {
        env_logger::init();
        // let d = test_detection_resampling2("testing_data/test_48khz_f32_mono.wav", 48000, 512, 1, OpenWakeWordHugo).unwrap();
        // assert!(d > 0.0);

        let handles = [
            // FIXME
            thread::spawn(move || test_detection_resampling3("testing_data/test_16khz_i16_mono.wav", 16000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_44khz_i16_mono.wav", 44100, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_44khz_f32_mono.wav", 44100, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_i16_mono.wav", 48000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_i16_mono.wav", 48000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_i16_mono.wav", 48000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_f32_mono.wav", 48000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_f32_stereo.wav", 48000, 512, 2, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_i16_stereo.wav", 48000, 512, 2, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_48khz_f32_stereo_one_channel.wav", 48000, 512, 2, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/rhino_puncochar_100_250313-082409_24071.wav", 16000, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_44khz_f32_mono.wav", 44100, 512, 1, OpenWakeWordHugo)),
            thread::spawn(move || test_detection_resampling3("testing_data/test_44khz_f32_stereo.wav", 44100, 512, 2, OpenWakeWordHugo)),
        ];

        let results_vec: Vec<f32> = handles
            .into_iter() // Use into_iter() to consume the array
            .map(|handle| handle.join().unwrap().unwrap())
            .collect();
        let results = results_vec.as_slice();

        for i in 1..results.len() {
            assert_relative_eq!(results[0], results[i], epsilon = 0.1);
        }
    }

    // #[ignore]
    // fn test_detection_with_resampling_rhino() {
    //     let _ = env_logger::try_init();
    //
    //     let handles = [
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_16khz_i16_mono.wav", 16000, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_44khz_i16_mono.wav", 44100, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_48khz_i16_mono.wav", 48000, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_48khz_i16_mono.wav", 48000, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_48khz_i16_mono.wav", 48000, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_48khz_f32_mono.wav", 48000, 1280, 1, PvRhino)),
    //         thread::spawn(move || test_detection_resampling2("testing_data/test_48khz_f32_stereo.wav", 48000, 1280, 2, PvRhino)),
    //     ];
    //
    //     let results_vec: Vec<f32> = handles
    //         .into_iter() // Use into_iter() to consume the array
    //         .map(|handle| handle.join().unwrap().unwrap())
    //         .collect();
    //     let results = results_vec.as_slice();
    //
    //     for i in 1..results.len() {
    //         assert_relative_eq!(results[0], results[i], epsilon = 0.1);
    //     }
    // // }
    //
    #[warn(dead_code)] // not really dead but used in tests
    fn test_detection_resampling3(filename: &str, original_sample_rate: usize, chunk_size: usize, channels: usize, model_type: SpeechUnlockType) -> Option<f32> {
        let buffer = Arc::new(Mutex::new(vec![]));

        let config = UnlockConfig {
            unlock_type: SpeechUnlockType::OpenWakeWordAlexa,
            yelling_threshold: 5000,
            detection_threshold: 0.5,
            quite_threshold: 10,
            endpoint_duration_secs: 0.5,
            save_wavs: false,
            unlock_time_secs: 30,
        };

        let pv_token = Some("---".to_string());

        let model1 = match model_type {
            OpenWakeWordAlexa => new_model(config.clone()).unwrap(),
            _ => panic!("Unsupported model type"),
        };

        let model2 = match model_type {
            OpenWakeWordAlexa => new_model(config.clone()).unwrap(),
            _ => panic!("Unsupported model type"),
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
                if let Ok(mut ring_buffer) = ring_buffer.lock() {
                    let data: Vec<i16> = ring_buffer.to_vec().iter().map(f32_to_i16).collect();
                    // save_full_wav_with_channels(&data, channels, MODEL_SAMPLE_RATE, prb.unwrap() as _, "me", "me", 0, false, ".");
                }
                return prb.map(|v| v as f32 / 100.0);
            }
            // sleep(Duration::from_millis(80));
        }
        if let Ok(mut ring_buffer) = ring_buffer.lock() {
            let data: Vec<i16> = ring_buffer.to_vec().iter().map(f32_to_i16).collect();
            // save_full_wav_with_channels(&data, channels, MODEL_SAMPLE_RATE, 0, "me", "me", 0, false, ".");
        }

        debug!("No detection in file {} and model {:?}", filename, model_type);
        None
    }
}
