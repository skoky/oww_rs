import wave

import numpy as np

from model import Model
from utils import AudioFeatures, CHUNK


def test_mel_spectogram():
    input_data = [0] * 64000
    mels = AudioFeatures()._get_melspectrogram(input_data)
    assert mels.shape == (397, 32)
    assert mels.max() == -7.999999
    assert mels.min() == -7.999999


def test_embeddings():
    input_data = np.zeros(64000).astype(np.int16)
    embeddings = AudioFeatures()._get_embeddings(input_data)
    assert embeddings.shape == (41, 96)
    assert embeddings.max() == 38.666107
    assert embeddings.min() == -37.116394


def test_audio_preproc():
    a = AudioFeatures()
    input_data = np.zeros(CHUNK).astype(np.int16)
    result = a(input_data)
    assert result
    assert a.feature_buffer.shape == (42, 96)
    assert a.melspectrogram_buffer.shape == (81, 32)


def test_model():
    model = Model(["hey_jarvis_v0.1.onnx"])
    input_data = np.zeros(CHUNK).astype(np.int16)
    predictions = model.predict(input_data)
    assert predictions['hey_jarvis_v0.1'] == 0.0


def test_model_less_data():
    model = Model(["hey_jarvis_v0.1.onnx"])
    input_data = np.zeros(CHUNK - 10).astype(np.int16)
    predictions = model.predict(input_data)
    assert predictions['hey_jarvis_v0.1'] == 0.0


def test_detection_from_wav():
    filedata = read_wav_int16('hey_jarvis.wav')
    model = Model(["hey_jarvis_v0.1.onnx"])
    chunks = np.array_split(filedata, np.ceil(len(filedata) / CHUNK))
    positive_predictions_count = 0
    for chunk in chunks:
        prediction = model.predict(chunk)
        p = prediction['hey_jarvis_v0.1']
        if p > 0.9:
            print(prediction)
            positive_predictions_count += 1

    assert 3 <= positive_predictions_count <= 5


def read_wav_int16(wav_file) -> np.ndarray:
    with wave.open(wav_file, 'rb') as wf:
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return samples
