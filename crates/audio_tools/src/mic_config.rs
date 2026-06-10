use crate::{SAMPLE_RATE_FOR_NOISE_REDUCTION, VOICE_SAMPLE_RATE};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, StreamConfig, SupportedStreamConfigRange};
use log::{debug, error, info, trace};
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

    let desired_config = supported_configs.iter().find(|config| {
        config.channels() == 1
            && config.sample_format() == SampleFormat::F32
            && config.min_sample_rate() <= desired_sample_rate
            && config.max_sample_rate() >= desired_sample_rate
    });

    if let Some(config_range) = desired_config {
        let config = config_range.with_sample_rate(desired_sample_rate);
        debug!("Found desired configuration (mono, f32, {desired_sample_rate} Hz)");
        return Ok((config.clone().into(), config_range.sample_format()));
    }

    // Try to find any mono configuration with f32
    let mono_f32_config = supported_configs
        .iter()
        .find(|config| config.channels() == 1 && config.sample_format() == SampleFormat::F32);

    if let Some(config_range) = mono_f32_config {
        let config = config_range.with_max_sample_rate();
        debug!("Found mono f32 configuration with different sample rate {:?}", &config);
        return Ok((config.into(), config_range.sample_format()));
    }

    // Try to find any stereo configuration with f32
    let stereo_f32_config = supported_configs
        .iter()
        .find(|config| config.channels() == 2 && config.sample_format() == SampleFormat::F32);

    if let Some(config_range) = stereo_f32_config {
        let config = config_range.with_max_sample_rate();
        debug!("Found stereo f32 configuration with different sample rate {:?}", &config);
        return Ok((config.into(), config_range.sample_format()));
    }

    // Try to find any mono configuration with the desired sample rate
    let mono_desired_rate_config = supported_configs
        .iter()
        .find(|config| config.channels() == 1 && config.min_sample_rate() <= desired_sample_rate && config.max_sample_rate() >= desired_sample_rate);

    if let Some(config_range) = mono_desired_rate_config {
        let config = config_range.with_max_sample_rate();
        debug!("Found mono configuration with different format but good sample rate {:?}", &config);
        return Ok((config.into(), config_range.sample_format()));
    }

    // Try to find any mono configuration
    let mono_config = supported_configs.iter().find(|config| config.channels() == 1);

    if let Some(config_range) = mono_config {
        let config = config_range.with_max_sample_rate();
        debug!("Found mono configuration with different format {:?}", &config);
        return Ok((config.into(), config_range.sample_format()));
    }

    let default_config = device.default_input_config()?;
    debug!("Using default configuration {:?} ", default_config);
    Ok((default_config.clone().into(), default_config.sample_format()))
}
