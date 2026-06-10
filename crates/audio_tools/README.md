# audio_tools

[![Crates.io](https://img.shields.io/crates/v/audio_tools.svg)](https://crates.io/crates/audio_tools)
[![Docs.rs](https://docs.rs/audio_tools/badge.svg)](https://docs.rs/audio_tools)
[![Build & test](https://github.com/skoky/oww_rs/actions/workflows/rust.yml/badge.svg)](https://github.com/skoky/oww_rs/actions/workflows/rust.yml)

Reusable, **model-agnostic** audio building blocks for a 16 kHz mono voice
pipeline. Extracted from [`oww-rs`](https://crates.io/crates/oww-rs) so it can be
consumed on its own by any project that needs to capture a microphone and feed
fixed-size, fixed-rate chunks to a speech model (wake-word, ASR, VAD, …).

It has **no dependency on any model or ML runtime** — just `cpal`, `rubato`,
`hound` and friends.

## Features

- **Microphone selection** (`mic_config`) — pick the best `cpal` input config
  for the active host, preferring mono `f32`. Optionally bias the search toward
  48 kHz when you intend to run noise reduction first.
- **FFT resampling** (`resampler`) — `make_resampler` builds a `rubato`
  resampler chain that converts any mic rate (44.1 kHz, 48 kHz, …) down to
  16 kHz, transparently using a two-stage upsample→downsample path for
  non-integer ratios.
- **Chunking** (`process_audio`) — `resample_into_chunks` buffers a raw `cpal`
  callback slice, resamples it and slices it into model-ready `Chunk`s, each
  carrying `f32` data, `i16` data and a precomputed RMS.
- **Channel handling** (`chunk`) — `ChannelsVec` / `Chunk` containers that hold
  1 or 2 channels, with interleave/deinterleave helpers.
- **Conversions** (`converters`) — branchless `i16 ↔ f32` sample conversion.
- **RMS** (`rms`) — per-chunk and windowed (circular-buffer) loudness, handy for
  a "is anyone speaking" gate.
- **WAV saving** (`save`) — write a ring buffer of chunks to a timestamped
  16-bit WAV (e.g. to capture detections for later inspection).
- **Denoising** (`denoiser`) — optional `nnnoiseless` RNNoise pass that you can
  run before resampling.

## Pipeline constants

`lib.rs` exposes the constants the whole pipeline shares —
`VOICE_SAMPLE_RATE` (16 kHz), `MODEL_SAMPLE_RATE`, `SAMPLE_RATE_FOR_NOISE_REDUCTION`
(48 kHz), `RING_BUFFER_SIZE`, `RMS_BUFFER_SIZE`, and more.

## Example

A runnable mic → 16 kHz chunk demo lives in
[`examples/mic_to_chunks.rs`](examples/mic_to_chunks.rs):

```bash
cargo run -p audio_tools --example mic_to_chunks
```

It opens the default input device, resamples whatever the mic produces down to
16 kHz mono, and prints the rate of 1280-sample (80 ms) chunks together with
their RMS level.

### Sketch

```rust
use audio_tools::mic_config::{default_input_device, find_best_config};
use audio_tools::process_audio::resample_into_chunks;
use audio_tools::resampler::make_resampler;
use std::sync::{Arc, Mutex};

// The fixed chunk size your downstream model expects (samples @ 16 kHz).
const MODEL_CHUNK_SIZE: u32 = 1280;

let device = default_input_device()?;
let (config, _sample_format) = find_best_config(&device, false)?;

let channels = config.channels as usize;
let mut resampler = make_resampler(config.sample_rate, MODEL_CHUNK_SIZE, channels)?;
let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

// Inside your cpal input callback, for each `data: &[f32]`:
let chunks = resample_into_chunks(data, &buffer, channels, &mut resampler);
for chunk in chunks {
    // chunk.data_f32.first() -> &Vec<f32> of MODEL_CHUNK_SIZE samples @ 16 kHz
    // chunk.rms             -> i16 loudness
}
```

## License

MIT — see [LICENSE](../../LICENSE).
