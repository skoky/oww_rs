use crate::config::UnlockConfig;
use crate::mic::RING_BUFFER_SIZE;
use crate::mic::converters::i16_to_f32;
use crate::mic::mic_config::find_best_config;
use crate::mic::process_audio::resample_into_chunks;
use crate::mic::resampler::make_resampler;
use crate::model::{Detection};
use crate::Models;
use circular_buffer::CircularBuffer;
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, error, info, warn};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use crate::chunk::{Chunk, ChunkType};
use std::sync::mpsc;

pub(crate) const MODEL_SAMPLE_RATE: u32 = 16000;

pub struct MicHandlerCpal {
    model: Arc<Mutex<Models>>,
    unlock_config: UnlockConfig,
    chunks_sender: broadcast::Sender<ChunkType>,
}

impl MicHandlerCpal {
    pub fn default_mic_name() -> Result<String, String> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or("No mic available")?;
        device.description().map(|d| d.name().to_string()).map_err(|e| e.to_string())
    }

    pub(crate) fn new(
        model: Arc<Mutex<Models>>,
        unlock_config: &UnlockConfig,
        chunks_sender: broadcast::Sender<ChunkType>,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(MicHandlerCpal {
            model,
            unlock_config: unlock_config.clone(),
            chunks_sender,
        })
    }

    pub fn loop_now_sync(&mut self, cancellation_token: CancellationToken) -> Result<bool, Box<dyn Error>> {
        // Initialize CPAL
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            None => return Err("No input device available".into()),
            Some(mic) => mic,
        };
        let mut last_mic_name = "".to_string();
        match device.description() {
            Ok(name) => {
                debug!("Input device: {}", name);
            }
            Err(e) => {
                error!("Couldn't get mic device: {:?}", e);
                return Err(e.to_string().into());
            }
        }
        info!("Starting to find the best config");
        let (config, sample_format) = match find_best_config(&device) {
            Ok(c) => c,
            Err(e) => {
                warn!("Mic not compatible {}", e);
                return Err(e.to_string().into());
            }
        };
        info!("Selected input config: {:?}", config);

        // Create a buffer to store audio data
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![]));
        let buffer_clone = buffer.clone();

        // Store the original sample rate and channels
        let original_sample_rate = config.sample_rate;
        let channels = config.channels as usize;

        // Create the input stream
        let err_fn = |err| warn!("An error occurred on the input stream: {}", err);

        let model_chunk_size = self.model.lock().unwrap().frame_length();

        let mut resamplers = make_resampler(original_sample_rate, model_chunk_size as _, channels)?;

        let (tx, rx) = mpsc::sync_channel::<Chunk>(100);
        
        let model_worker = self.model.clone();
        let chunks_sender_worker = self.chunks_sender.clone();
        let quite_threshold = self.unlock_config.quite_threshold;
        let cancellation_token_worker = cancellation_token.clone();

        spawn(move || {
            let mut ring_buffer = CircularBuffer::<{ RING_BUFFER_SIZE }, Chunk>::boxed();
            while !cancellation_token_worker.is_cancelled() {
                if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(100)) {
                    ring_buffer.push_back(chunk.clone());
                    let detection = handle_detect(
                        &chunk,
                        chunks_sender_worker.clone(),
                        model_worker.clone(),
                    );
                    if detection.detected {
                        let sum_rms: u64 = ring_buffer.iter().map(|c| c.rms.abs() as u64).sum();
                        let avg_rms = (sum_rms / ring_buffer.len() as u64) as i16;
                        println!("Detection {:?}, avg_rms {} limit {}", detection, avg_rms, quite_threshold);
                    }
                }
            }
            debug!("Model worker thread finished");
        });

        let timeout = Some(Duration::from_millis(80));

        let tx = tx.clone();
        let stream = build_input_stream(
            &device,
            &config,
            sample_format,
            move |data_f32| {
                resample_into_chunks(data_f32, &buffer_clone, channels, &mut resamplers).into_iter().for_each(|chunk| {
                    if let Err(_) = tx.try_send(chunk) {
                        warn!("Model worker channel full, dropping chunk");
                    }
                })
            },
            err_fn,
            timeout,
        )?;

        stream.play()?;

        let mut check_mic_count: u64 = 0;
        while !cancellation_token.is_cancelled() {
            check_mic_count += 1;
            if check_mic_count % 20 == 0 {
                let mic_name = self.check_mic();
                match mic_name {
                    Some(mic_name) if last_mic_name != mic_name => {
                        last_mic_name = mic_name.clone();
                    }
                    None => {
                        // mic changed, reload recorder
                        break;
                    }
                    _ => {}
                }
            }

            sleep(Duration::from_millis(100));
        }
        Ok(false)
    }

    fn check_mic(&self) -> Option<String> {
        match MicHandlerCpal::default_mic_name() {
            Ok(default_mic) => Some(default_mic),
            Err(e) => {
                warn!("Mic error {}", e);
                None
            }
        }
    }
}

pub fn build_input_stream<F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: SampleFormat,
    mut process_data: F,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    timeout: Option<Duration>,
) -> Result<cpal::Stream, Box<dyn Error>>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| process_data(data),
            err_fn,
            timeout,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let data_f32: Vec<f32> = data.iter().map(i16_to_f32).collect();
                process_data(&data_f32);
            },
            err_fn,
            timeout,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };
    Ok(stream)
}

pub(crate) fn handle_detect(
    chunk: &Chunk,
    chunks_sender: broadcast::Sender<ChunkType>,
    model: Arc<Mutex<Models>>,
) -> Detection {
    let _ = chunks_sender.send(chunk.clone());

    let mut model = model.lock().unwrap();
    let mut detections_f32 = vec![];

    for (channel, data) in chunk.data_f32.iter().enumerate() {
        let d = if channel == 0 { model.detect1(data.clone()) } else { model.detect2(data.clone()) };
        #[cfg(debug_assertions)]
        d.clone().map(|d| {
            if d.probability > 0.0 {
                debug!("Detection channel {}, prob {}", channel, d.probability)
            }
        });
        detections_f32.push(d);
    }
    let detections_i16: Vec<Option<Detection>> = chunk
        .data_i16
        .iter()
        .enumerate()
        .map(|(i, data)| if i == 0 { model.detect1_i16(data.clone()) } else { model.detect2_i16(data.clone()) })
        .collect();
    // find the best detection
    let detection = detections_f32
        .into_iter()
        .chain(detections_i16) // flatten both vectors into one iterator
        .flatten() // remove the None values
        .max_by(|d1, d2| d1.probability.partial_cmp(&d2.probability).unwrap())
        .unwrap();

    detection
}
