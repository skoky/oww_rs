use crate::{SAMPLE_RATE_FOR_NOISE_REDUCTION, VOICE_SAMPLE_RATE};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, StreamConfig, SupportedStreamConfig, SupportedStreamConfigRange};
use log::{debug, error, info, trace, warn};
use std::error::Error;
use std::io;

pub fn default_input_device() -> Result<cpal::Device, Box<dyn Error>> {
    cpal::default_host()
        .default_input_device()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Could not find input device").into())
}

/// Finds the best mic config from those available. If noise reduction is enabled, search for 48 kHz/f32. If not enabled
/// 16 kHz/f32 is sufficient for the speech model. The 48 kHz is resampled to 16 kHz later anyway.
pub fn find_best_config(device: &cpal::Device, optimal_for_noise_reduction: bool) -> Result<(StreamConfig, SampleFormat), Box<dyn Error>> {
    #[cfg(target_os = "ios")]
    {
        let config = match device.default_input_config() {
            Ok(config) => {
                info!("IOS Mic config: {:?}", config);
                config
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        return Ok((config.clone().into(), config.sample_format()));
    }

    let supported_configs: Vec<SupportedStreamConfigRange> = match device.supported_input_configs() {
        Ok(supported_configs) => {
            let supported_configs: Vec<SupportedStreamConfigRange> = supported_configs.collect();
            info!("found {} supported input configs", supported_configs.len());
            supported_configs
        }
        Err(e) => {
            error!("error getting supported_input_configs : {:?}", e);
            return Err(Box::new(e));
        }
    };

    if supported_configs.is_empty() {
        let device_name = device
            .description()
            .map(|description| format!("{description:?}"))
            .unwrap_or_else(|_| "<unknown input device>".to_string());
        error!("input device `{device_name}` exposes no supported input configs; microphone is likely missing");
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("input device `{device_name}` has no supported input configs"),
        )
        .into());
    }

    for config in &supported_configs {
        trace!("Supported input config: {:?}", config);
    }

    // Prefer 16 kHz mono f32 (or 48 kHz when optimizing for noise reduction)
    let desired_sample_rate = if optimal_for_noise_reduction {
        SAMPLE_RATE_FOR_NOISE_REDUCTION as u32
    } else {
        VOICE_SAMPLE_RATE as u32
    };

    let default_config = device.default_input_config();
    if let Err(e) = &default_config {
        debug!("no default input config: {e}");
    }
    match pick_config(&supported_configs, default_config.as_ref().ok(), desired_sample_rate) {
        Some(config) => Ok((config.config(), config.sample_format())),
        None => {
            let default_config = default_config?;
            Ok((default_config.config(), default_config.sample_format()))
        }
    }
}

fn is_mono_f32(config: &SupportedStreamConfigRange) -> bool {
    config.channels() == 1 && config.sample_format() == SampleFormat::F32
}

fn is_stereo_f32(config: &SupportedStreamConfigRange) -> bool {
    config.channels() == 2 && config.sample_format() == SampleFormat::F32
}

fn is_mono(config: &SupportedStreamConfigRange) -> bool {
    config.channels() == 1
}

/// Channel/format classes in order of preference, each with a name for the log.
type ConfigClass = (&'static str, fn(&SupportedStreamConfigRange) -> bool);
const CONFIG_CLASSES: [ConfigClass; 3] = [("mono f32", is_mono_f32), ("stereo f32", is_stereo_f32), ("mono", is_mono)];

/// Picks a capture config from the device's ranges, in this order:
/// 1. mono f32 at exactly `desired_sample_rate`;
/// 2. mono f32, then stereo f32, then mono of any format, at a rate of at least [`VOICE_SAMPLE_RATE`]:
///    a whole multiple of it first (resamples in one stage), then the one closest to `desired_sample_rate`;
/// 3. the device default config, if it is at least [`VOICE_SAMPLE_RATE`];
/// 4. only then a rate below [`VOICE_SAMPLE_RATE`] (same channel/format order, highest rate), and
///    finally the device default whatever its rate.
///
/// Rates below 16 kHz are useless for speech: some Windows drivers list a 5512 Hz range first,
/// which the old "first mono f32 range at its max rate" fallback picked.
fn pick_config(
    supported_configs: &[SupportedStreamConfigRange],
    default_config: Option<&SupportedStreamConfig>,
    desired_sample_rate: u32,
) -> Option<SupportedStreamConfig> {
    let min_rate = VOICE_SAMPLE_RATE as u32;

    if let Some(config) = supported_configs
        .iter()
        .filter(|config| is_mono_f32(config))
        .find_map(|config| config.try_with_sample_rate(desired_sample_rate))
    {
        debug!("Found desired configuration (mono, f32, {desired_sample_rate} Hz)");
        return Some(config);
    }

    for (name, class) in CONFIG_CLASSES {
        let speech_rate = supported_configs
            .iter()
            .filter(|config| class(config) && config.max_sample_rate() >= min_rate)
            .map(|config| {
                let low = config.min_sample_rate().max(min_rate);
                let rate = desired_sample_rate.clamp(low, config.max_sample_rate());
                // A range that skips the desired rate may still hold a whole multiple of 16 kHz.
                let multiple = low.div_ceil(min_rate) * min_rate;
                let rate = if rate % min_rate != 0 && multiple <= config.max_sample_rate() { multiple } else { rate };
                (*config).with_sample_rate(rate)
            })
            .min_by_key(|config| (config.sample_rate() % min_rate != 0, config.sample_rate().abs_diff(desired_sample_rate)));
        if let Some(config) = speech_rate {
            debug!("Found {name} configuration with different sample rate {config:?}");
            return Some(config);
        }
    }

    if let Some(config) = default_config.filter(|config| config.sample_rate() >= min_rate) {
        debug!("Using default configuration {:?} ", config);
        return Some(config.clone());
    }

    for (name, class) in CONFIG_CLASSES {
        let low_rate = supported_configs
            .iter()
            .filter(|config| class(config))
            .max_by_key(|config| config.max_sample_rate())
            .map(|config| (*config).with_max_sample_rate());
        if let Some(config) = low_rate {
            warn!("Only a low sample rate is available: {name} {config:?}; speech recognition will suffer");
            return Some(config);
        }
    }

    if let Some(config) = default_config {
        warn!("Using default configuration with a low sample rate {:?} ", config);
    }
    default_config.cloned()
}

#[cfg(test)]
mod tests {
    use super::pick_config;
    use cpal::{SampleFormat, SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange};

    fn range(channels: u16, min: u32, max: u32, format: SampleFormat) -> SupportedStreamConfigRange {
        SupportedStreamConfigRange::new(channels, min, max, SupportedBufferSize::Unknown, format)
    }

    fn rate_of(config: Option<SupportedStreamConfig>) -> (u16, u32, SampleFormat) {
        let config = config.expect("a config");
        (config.channels(), config.sample_rate(), config.sample_format())
    }

    #[test]
    fn test_exact_mono_f32() {
        let ranges = [range(1, 8000, 48000, SampleFormat::F32)];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (1, 16000, SampleFormat::F32));
    }

    #[test]
    fn test_skips_low_rate_mono_f32() {
        // Windows 11 laptop: discrete rates listed as separate ranges, 5512 Hz first.
        let ranges = [
            range(1, 5512, 5512, SampleFormat::F32),
            range(1, 11025, 11025, SampleFormat::F32),
            range(1, 44100, 44100, SampleFormat::F32),
            range(1, 48000, 48000, SampleFormat::F32),
        ];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (1, 48000, SampleFormat::F32));
        assert_eq!(rate_of(pick_config(&ranges, None, 48000)), (1, 48000, SampleFormat::F32));
    }

    #[test]
    fn test_prefers_whole_multiple_of_16khz() {
        let ranges = [range(1, 22050, 22050, SampleFormat::F32), range(1, 44100, 96000, SampleFormat::F32)];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (1, 48000, SampleFormat::F32));
        // Without one, the rate nearest the desired one.
        let ranges = [range(1, 44100, 44100, SampleFormat::F32), range(1, 22050, 22050, SampleFormat::F32)];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (1, 22050, SampleFormat::F32));
    }

    #[test]
    fn test_prefers_stereo_speech_rate_over_low_mono() {
        let ranges = [range(1, 5512, 5512, SampleFormat::F32), range(2, 48000, 48000, SampleFormat::F32)];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (2, 48000, SampleFormat::F32));
    }

    #[test]
    fn test_prefers_default_over_low_rate() {
        let ranges = [range(1, 5512, 5512, SampleFormat::F32), range(2, 8000, 11025, SampleFormat::I16)];
        let default = range(2, 48000, 48000, SampleFormat::I16).with_max_sample_rate();
        assert_eq!(rate_of(pick_config(&ranges, Some(&default), 16000)), (2, 48000, SampleFormat::I16));
    }

    #[test]
    fn test_low_rate_as_last_resort() {
        let ranges = [range(1, 5512, 5512, SampleFormat::F32), range(1, 8000, 11025, SampleFormat::F32)];
        assert_eq!(rate_of(pick_config(&ranges, None, 16000)), (1, 11025, SampleFormat::F32));
        let default = range(1, 8000, 8000, SampleFormat::I16).with_max_sample_rate();
        assert_eq!(rate_of(pick_config(&[], Some(&default), 16000)), (1, 8000, SampleFormat::I16));
    }
}
