# oww-rs

[![Crates.io](https://img.shields.io/crates/v/oww-rs.svg)](https://crates.io/crates/oww-rs)
[![Docs.rs](https://docs.rs/oww-rs/badge.svg)](https://docs.rs/oww-rs)
[![Build & test](https://github.com/skoky/oww_rs/actions/workflows/rust.yml/badge.svg)](https://github.com/skoky/oww_rs/actions/workflows/rust.yml)

A minimal, **inference-only** Rust port of
[openWakeWord](https://github.com/dscripka/openWakeWord). It reimplements the
ONNX inference path — the shared *melspectrogram → speech-embedding →
wakeword-classifier* pipeline — for low-latency wake-word detection that is easy
to embed in a Rust application.

There is **no training here**: models are trained with the upstream Python
project and consumed as `.onnx` files. The runtime is
[`tract-onnx`](https://crates.io/crates/tract-onnx), so there is no native ONNX
Runtime dependency to ship.

## Features

- **Self-contained models** — the melspectrogram + embedding front-end and the
  per-wakeword classifiers are embedded into the binary at compile time
  (`rust-embed`). Ships with `alexa` and `hey_mycroft` out of the box.
- **openWakeWord-compatible front-end** — 16 kHz mono `f32` audio in 1280-sample
  (80 ms) chunks, melspectrogram normalization and tensor shapes mirror the
  Python implementation so detection behaviour matches upstream.
- **Falling-edge detection smoothing** — fires on the falling edge of the
  probability curve with a refractory window, which is more robust than a naive
  threshold crossing.
- **Microphone capture loop** — a `cpal`-based capture/auto-reconnect loop
  (`create_unlock_task_sync`) that resamples any mic to 16 kHz and runs detection.
- **Built on [`audio_tools`](https://crates.io/crates/audio_tools)** for mic
  config, resampling and chunking.

## Quick start

Add the crate:

```toml
[dependencies]
oww-rs = "0.3"
```

Run inference over an audio stream, chunk by chunk:

```rust
use oww_rs::oww::{OwwModel, OWW_MODEL_CHUNK_SIZE};
use oww_rs::config::SpeechUnlockType::OpenWakeWordAlexa;

// threshold 0.1 — detection logic smooths over a ~1 s window
let mut model = OwwModel::new(OpenWakeWordAlexa, 0.1)?;

// Feed 16 kHz mono f32 chunks of OWW_MODEL_CHUNK_SIZE (1280) samples:
for chunk in audio_chunks {
    let detection = model.detection(chunk);
    if detection.detected {
        println!("wake word! probability = {}", detection.probability);
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Live microphone demo

```bash
cargo run -p oww-rs --example cpal_test   # then say "Alexa"
```

The example ([`examples/cpal_test.rs`](examples/cpal_test.rs)) opens the default
mic, resamples it to 16 kHz via `audio_tools`, and prints a line whenever the
wake word fires.

## Adding a new wake word

1. Train a classifier with the upstream openWakeWord project and export it to
   `.onnx`.
2. Drop the file into `speech_models/`.
3. Add a variant to `SpeechUnlockType` (`config.rs`) and wire it into
   `OwwModel::new` / `new_model`.

See the upstream [openWakeWord](https://github.com/dscripka/openWakeWord)
project for pre-trained models and training instructions.

## License

MIT — see [LICENSE](../../LICENSE).
