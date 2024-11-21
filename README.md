# Minimalistic version of the OpenWakeWord inference in Rust and ONNX runtime

This is extracted source code from the [OpenWakeWord](https://github.com/dscripka/openWakeWord) 
providing simplified code for inference of onnx models only. The Python code for inference is converted to
Rust for better wake word detection performance. The inference uses [ORT](https://github.com/pykeio/ort)
supporting all major platforms. For detection the `hey_jarvis_v0.1.onnx` model is used as a sample from
the OpenWakeWord project. There is also `hey_jarvis.wav` file used in test to trigger wake word detection with 
more than 99% accuracy.

The implementation uses [pv_recorded](https://github.com/Picovoice/pvrecorder) opensourced audio recorder from
[Picovoice](www.picovoice.com). The recorded uses underlying [miniaudio](https://github.com/mackron/miniaudio) 
lib for accessing microphones on all major platforms. 

See the original [OpenWakeWord](https://github.com/dscripka/openWakeWord) lib for more details for pre-trained models or training new custom model. The
training stage is not included in this repo and the OpenWakeWord's original steps should be used to generate
new model triggering on a custom wake word.

If you are looking for simplified Python version, see git log with tag `python_version`.

# Running like this

    cargo run --example from_mic

or tokio based version

    cargo run --example from_mic_tokio

Running tests:

    cargo test

# Contribution

Make an issue / PR etc.

# Licence

MIT license, see [LICENSE](./LICENSE)

