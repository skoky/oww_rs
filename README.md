[![Build & test](https://github.com/skoky/oww_rs/actions/workflows/rust.yml/badge.svg)](https://github.com/skoky/oww_rs/actions/workflows/rust.yml)

# oww-rs — minimalistic OpenWakeWord inference in Rust

This is extracted source code from [openWakeWord](https://github.com/dscripka/openWakeWord),
providing simplified code for **inference of ONNX models only**. The Python inference path
is converted to Rust for better wake-word detection performance. Detection runs on the
[`tract-onnx`](https://github.com/sonos/tract) runtime, supporting all major platforms.
The `alexa.onnx` model is bundled as a sample — simply saying "Alexa" into the microphone
triggers a detection, as implemented in `crates/oww/examples/cpal_test.rs`.

Microphone access uses the pure-Rust [`cpal`](https://github.com/RustAudio/cpal) crate,
which works on all major platforms.

See the original [openWakeWord](https://github.com/dscripka/openWakeWord) project for
pre-trained models and for training new custom wake words. **Training is not included
here** — use openWakeWord's steps to generate a model for a custom wake word.

## Workspace layout

This is a Cargo workspace with two publishable crates:

| Crate | Published as | What it is |
|-------|--------------|------------|
| [`crates/audio_tools`](crates/audio_tools) | [`audio_tools`](https://crates.io/crates/audio_tools) | Reusable, **model-agnostic** mic capture, resampling, channel handling, conversion, RMS and WAV saving for a 16 kHz mono pipeline. |
| [`crates/oww`](crates/oww) | [`oww-rs`](https://crates.io/crates/oww-rs) | The wake-word ONNX inference (tract) + mic-capture detection loop. Depends on `audio_tools`. |

## Running

```bash
cargo run -p oww-rs --example cpal_test            # live mic demo — say "Alexa"
cargo run -p audio_tools --example mic_to_chunks   # mic → 16 kHz chunk demo
cargo test                                         # run the whole workspace
```

## Contribution

Open an issue / PR.

## License

MIT license, see [LICENSE](./LICENSE)
