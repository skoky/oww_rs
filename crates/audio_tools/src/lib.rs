//! Reusable audio-processing building blocks for the 16 kHz mono wake-word pipeline:
//! microphone configuration, FFT resampling, channel handling, i16/f32 conversion,
//! RMS and WAV saving. This crate has no dependency on the wake-word model and can be
//! consumed standalone.

pub mod chunk;
pub mod converters;
pub mod denoiser;
pub mod mic_config;
pub mod process_audio;
pub mod resampler;
pub mod rms;
pub mod save;

/// Sample rate the wake-word model and the whole pipeline operate at.
pub const VOICE_SAMPLE_RATE: usize = 16000;
/// Same value as a `u32`, used by the resampler / mic config.
pub const MODEL_SAMPLE_RATE: u32 = 16000;
/// Capture rate preferred when noise reduction is enabled (resampled down to 16 kHz later).
pub const SAMPLE_RATE_FOR_NOISE_REDUCTION: usize = 48000;

/// Number of RMS samples kept (~1 s of history).
pub const RMS_BUFFER_SIZE: usize = 16;
/// Typical CPAL input callback chunk size.
pub const EXPECTED_CHUNK_SIZE: usize = 512;
/// Seconds of audio retained in the ring buffer.
pub const BUFFER_SECS: usize = 4;
/// Ring-buffer capacity in chunks.
pub const RING_BUFFER_SIZE: usize = (VOICE_SAMPLE_RATE * BUFFER_SECS) / EXPECTED_CHUNK_SIZE;
/// ~80 ms detection cadence threshold (chunk-size based).
pub const MAX_DETECTION_DURATION_SECS: u128 = 80;
