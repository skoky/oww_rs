# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`oww-rs` is a minimal **inference-only Rust port** of [openWakeWord](https://github.com/dscripka/openWakeWord) (the Python project lives at `../openWakeWord`). It reimplements the ONNX inference path — the shared melspectrogram → speech-embedding → wakeword-classifier pipeline — for better runtime performance and easy embedding into Rust apps. There is **no training here**: models are trained in the Python project and consumed as `.onnx` files.

Despite the README mentioning ORT/pykeio, the actual runtime is **`tract-onnx`** (see `crates/oww/Cargo.toml`).

## Workspace layout

This is a **Cargo workspace** (virtual manifest at the repo root) with two crates:

- **`crates/audio_tools`** (`audio_tools`) — reusable, model-agnostic audio building blocks: `cpal` mic config, `rubato` resampling, channel handling, i16/f32 conversion, RMS, WAV save, and the shared `Chunk` type + pipeline constants. **No dependency on the wake-word model** — meant to be consumed standalone (the goal is for the sibling `pnp` and `chytryasistent` repos to depend on this crate instead of carrying their own copies).
- **`crates/oww`** (package `oww-rs`, lib `oww_rs`) — the wake-word inference (tract) + the mic-capture loop that drives detection. Depends on `audio_tools`.

## Commands

```bash
cargo build                         # build both crates
cargo build --examples              # build the cpal demo
cargo run -p oww-rs --example cpal_test   # live mic demo — say "Alexa" to trigger
cargo test                          # run the whole workspace
cargo test -p audio_tools           # test just the audio crate
cargo test test_mels2               # run a single test by name
```

CI (`.github/workflows/rust.yml`) runs `cargo build` + `cargo test` on `ubuntu-latest`. Edition is **2024** (needs a recent stable toolchain).

## Inference architecture

The pipeline (in `crates/oww`) mirrors openWakeWord's Python front-end exactly, so the two must stay numerically in sync. Audio is **16 kHz mono f32, processed in 1280-sample (80 ms) chunks** (`OWW_MODEL_CHUNK_SIZE`).

1. **Embedded models** (`rust-embed`, baked into the binary at compile time; folders resolve relative to `crates/oww/`):
   - `crates/oww/models/` — the shared front-end: `melspectrogram.onnx` + `embedding_model.onnx` (loaded in `oww/audio.rs`).
   - `crates/oww/speech_models/` — per-wakeword classifiers, e.g. `alexa.onnx` (loaded in `oww/oww_model.rs`).
2. **Feature front-end** (`AudioFeaturesTract` in `crates/oww/src/oww/audio.rs`): a 1280-sample chunk → melspectrogram model → `[5,32]` mel frame, **rescaled `v/10 + 2`** (this matches openWakeWord's mel normalization). Mel frames are buffered, stacked to `[80,32]`, sliced to `[76,32]`, reshaped `[1,76,32,1]` → embedding model → `[1,1,1,96]` feature vector. Feature vectors are buffered 16 deep → `[16,96]`.
3. **Classifier** (`crates/oww/src/oww/oww_model.rs`): the `[16,96]` features reshape to `[1,16,96]` → wakeword model → `[1,1]` probability.
4. **Detection smoothing** — non-obvious, read before changing: per-frame probabilities go into a 12-deep buffer (~1 s). A detection does **not** fire when probability crosses the threshold; it fires on the **falling edge** — when the current probability drops below `0.1` while the running average of above-threshold frames is still high, gated by a 2 s refractory (`NO_DETECTION_MS`) and a minimum positive-frame count (`MIN_POSITIVE_DETECTIONS`). See `OwwModel::detect` / `calculate_average`.

## Module map

**`crates/audio_tools/src/`** (model-agnostic):
- `lib.rs` — pipeline constants (`VOICE_SAMPLE_RATE`, `MODEL_SAMPLE_RATE`, `RING_BUFFER_SIZE`, …) and module exports.
- `mic_config.rs` — `cpal` device selection / `find_best_config`.
- `resampler.rs`, `process_audio.rs` — `rubato` resampling to 16 kHz and chunking (`make_resampler`, `resample_into_chunks`).
- `converters.rs` — i16↔f32. `chunk.rs` — `ChannelsVec` / `Chunk` containers. `rms.rs` — RMS. `save.rs` — WAV writing.

**`crates/oww/src/`** (wake-word):
- `lib.rs` — crate root, `OWW_MODEL_CHUNK_SIZE`, the `Models` two-model wrapper, `load_wav` helpers, and `create_unlock_task_sync` (the long-running mic loop that auto-reconnects).
- `mic_cpal.rs` — the `cpal` capture loop that pulls chunks from `audio_tools` and runs detection (`MicHandlerCpal`, `handle_detect`). This is the integration glue and is **deliberately not** in `audio_tools` (it depends on `Models`).
- `model.rs` — the `Model` trait + `Detection` struct + `new_model` factory.
- `oww.rs`, `oww/audio.rs`, `oww/oww_model.rs` — the `OwwModel` implementation (front-end + classifier + detection logic above).
- `config.rs` — `UnlockConfig`; persistence via `confy` is commented out, only `Default` is live.

## Conventions

- Audio is always 16 kHz mono; feed the model 1280-sample (80 ms) f32 chunks. Mics are captured at native rate/format and resampled via `rubato` (see `crates/oww/examples/cpal_test.rs`).
- **Keep `audio_tools` model-free**: it must not depend on `oww-rs` or any tract/model type, so the sibling repos can consume it standalone. The resampler takes the model chunk size as a parameter rather than importing a wake-word constant.
- **Adding a new wakeword model**: drop the `.onnx` into `speech_models/`, add a variant to `SpeechUnlockType` (`config.rs`) and wire it in `OwwModel::new` and `new_model`.
- When the Python project's inference math changes (mel scaling, buffer sizes, tensor shapes), mirror it here — the two implementations share the same `.onnx` front-end and must agree.
