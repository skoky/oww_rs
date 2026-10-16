use crate::MODEL_SAMPLE_RATE;
use log::{debug, info, warn};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use std::error::Error;
use std::fmt::{Debug, Formatter};

const HIGH_SAMPLING_RATE: usize = 48_000;

pub struct Resamplers {
    pub main: Fft<f32>,
    pub upsampler: Fft<f32>,
    pub use_upsampler: bool,
    /// Input frames (per channel) the next [`resample_audio`] call consumes. Not constant: on the
    /// double-resampling path the `FixedSync::Output` upsampler wants a different block size from
    /// call to call (e.g. 689, 689, 689, 689, 0 at 5512 Hz), so read it again before every call.
    /// [`resample_audio`] keeps it up to date.
    pub chunk_for_resampling: usize,
    pub resample_rate: f32,
}

impl Resamplers {
    /// Input frames (per channel) the active resampler stage needs for its next block.
    pub fn input_frames_next(&self) -> usize {
        if self.use_upsampler { self.upsampler.input_frames_next() } else { self.main.input_frames_next() }
    }
}

impl Debug for Resamplers {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!(
            "use {}, chunk size {}, resample rate {}",
            self.use_upsampler, self.chunk_for_resampling, self.resample_rate
        ))
    }
}

pub fn make_resampler(original_sample_rate: u32, model_chunk_size: u32, channels_count: usize) -> Result<Resamplers, Box<dyn Error>> {
    let sample_rate = original_sample_rate as f32 / MODEL_SAMPLE_RATE as f32;

    if sample_rate.floor() != sample_rate {
        warn!("mic sample rate not supported {}; using double resampling", sample_rate);
        let internal_rate = HIGH_SAMPLING_RATE as f32 / MODEL_SAMPLE_RATE as f32;

        let main_sampler_input_size = (model_chunk_size as f32 * internal_rate) as usize;

        let main = Fft::<f32>::new(HIGH_SAMPLING_RATE, MODEL_SAMPLE_RATE as _, main_sampler_input_size, 1, channels_count, FixedSync::Input)?;

        let upsampler_output_size = (model_chunk_size as f32 * internal_rate) as usize;

        let upsampler = Fft::<f32>::new(original_sample_rate as _, HIGH_SAMPLING_RATE, upsampler_output_size, 1, channels_count, FixedSync::Output)?;
        // Taken from rubato, not computed from the rate ratio: the upsampler works in whole FFT
        // sub-chunks, so e.g. 5512 Hz -> 48 kHz needs 689 frames per block, not 4800 * 5512 / 48000.
        let upsampler_input_size = upsampler.input_frames_next();

        let resamplers = Resamplers {
            main,
            upsampler,
            use_upsampler: true,
            chunk_for_resampling: upsampler_input_size,
            resample_rate: sample_rate,
        };
        debug!("Resamplers: {:?}", resamplers);
        Ok(resamplers)
    } else {
        info!("mic sample rate {}", sample_rate);

        let input_chunk_size: usize = (model_chunk_size * sample_rate as u32) as usize;

        let upsampler = Fft::<f32>::new(original_sample_rate as _, MODEL_SAMPLE_RATE as _, input_chunk_size, 1, channels_count, FixedSync::Output)?;
        let main = Fft::<f32>::new(original_sample_rate as _, MODEL_SAMPLE_RATE as _, input_chunk_size, 1, channels_count, FixedSync::Input)?;

        let chunk_for_resampling = main.input_frames_next();

        Ok(Resamplers {
            main,
            upsampler,
            use_upsampler: false,
            chunk_for_resampling,
            resample_rate: sample_rate,
        })
    }
}

/// Resamples one block. `data` must hold `resamplers.chunk_for_resampling` frames (interleaved);
/// that count is refreshed after every call, so read it again before the next one.
pub fn resample_audio(data: &[f32], resamplers: &mut Resamplers, channels: usize) -> Result<Vec<f32>, String> {
    let frames = data.len() / channels;
    let input = InterleavedSlice::new(data, channels, frames).map_err(|e| format!("input adapter error: {}", e))?;

    let resampled = if resamplers.use_upsampler {
        let presampled = resamplers.upsampler.process(&input, 0, None).map_err(|e| format!("upsampler error: {}", e))?;
        resamplers.main.process(&presampled, 0, None).map_err(|e| format!("main resampler error: {}", e))?
    } else {
        resamplers.main.process(&input, 0, None).map_err(|e| format!("main resampler error: {}", e))?
    };
    resamplers.chunk_for_resampling = resamplers.input_frames_next();
    Ok(resampled.take_data())
}

#[cfg(test)]
pub mod tests {
    use crate::VOICE_SAMPLE_RATE;
    use crate::resampler::{make_resampler, resample_audio};

    /// Feeds `blocks` consecutive blocks of `chunk_for_resampling` frames through `resample_audio`
    /// and checks every block yields exactly one model chunk per channel.
    fn assert_resamples(sample_rate: u32, model_chunk_size: u32, channels: usize, blocks: usize) -> Vec<usize> {
        let mut r = make_resampler(sample_rate, model_chunk_size, channels).unwrap();
        let mut sizes = vec![];
        for _ in 0..blocks {
            let frames = r.chunk_for_resampling;
            sizes.push(frames);
            let input = vec![0.1_f32; frames * channels];
            let out = resample_audio(&input, &mut r, channels).unwrap_or_else(|e| panic!("{sample_rate} Hz, {frames} frames: {e}"));
            assert_eq!(out.len(), model_chunk_size as usize * channels, "{sample_rate} Hz output size");
        }
        sizes
    }

    #[test]
    fn test_mono_16khz() {
        let r = make_resampler(VOICE_SAMPLE_RATE as _, 1280, 1).unwrap();
        assert_eq!(r.resample_rate, 1.0);
        assert_eq!(r.chunk_for_resampling, 1280);
    }

    #[test]
    fn test_stereo_16khz() {
        let r = make_resampler(VOICE_SAMPLE_RATE as _, 1280, 2).unwrap();
        assert_eq!(r.resample_rate, 1.0);
        assert_eq!(r.chunk_for_resampling, 1280);
    }

    #[test]
    fn test_mono_48khz() {
        let r = make_resampler(48000, 1280, 1).unwrap();
        assert_eq!(r.resample_rate, 3.0);
        assert_eq!(r.chunk_for_resampling, 1280 * r.resample_rate as usize);
    }

    #[test]
    fn test_stereo_48khz() {
        let r = make_resampler(48000, 1280, 2).unwrap();
        assert_eq!(r.resample_rate, 3.0);
        assert_eq!(r.chunk_for_resampling, 3840);
    }

    #[test]
    fn test_mono_44_1khz() {
        let r = make_resampler(44100, 1280, 1).unwrap();
        assert_eq!(r.resample_rate, 2.75625);
        assert_eq!(r.chunk_for_resampling, 3528);
    }

    #[test]
    fn test_stereo_44_1khz() {
        let r = make_resampler(44100, 1280, 2).unwrap();
        assert_eq!(r.resample_rate, 2.75625);
        assert_eq!(r.chunk_for_resampling, 3528);
    }

    #[test]
    fn test_mono_5512hz_resamples() {
        let sizes = assert_resamples(5512, 1600, 1, 20);
        // FixedSync::Output upsampler: 689-frame FFT blocks, one in five calls drains saved output.
        assert_eq!(&sizes[..5], &[689, 689, 689, 689, 0]);
        // Long-run average matches the rate ratio: 1600 * 5512 / 16000 = 551.2 frames per chunk.
        assert_eq!(sizes.iter().sum::<usize>(), 20 * 5512 / 10);
    }

    #[test]
    fn test_stereo_5512hz_resamples() {
        assert_resamples(5512, 1600, 2, 20);
    }

    #[test]
    fn test_mono_11025hz_resamples() {
        let sizes = assert_resamples(11025, 1600, 1, 32);
        assert_eq!(sizes[0], 1176);
        assert_eq!(sizes.iter().sum::<usize>() as f64 / sizes.len() as f64, 1600.0 * 11025.0 / 16000.0);
    }

    #[test]
    fn test_stereo_11025hz_resamples() {
        assert_resamples(11025, 1600, 2, 32);
    }

    #[test]
    fn test_common_rates_resample() {
        for rate in [8000, 16000, 22050, 32000, 44100, 48000, 96000] {
            for chunk in [1280, 1600] {
                assert_resamples(rate, chunk, 1, 25);
            }
        }
    }
}
