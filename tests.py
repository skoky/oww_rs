import json
import os
import wave
from collections import deque

import numpy as np

from model import Model
from utils import AudioFeatures, CHUNK


def test_mel_spectogram():
    input_data = [0] * 64000
    mels = AudioFeatures()._get_melspectrogram(input_data)
    assert mels.shape == (397, 32)
    assert mels.max() == -7.999999
    assert mels.min() == -7.999999
    save_result(mels, "mels_from_empty_array.txt")


def test_embeddings():
    input_data = np.zeros(64000).astype(np.int16)
    embeddings = AudioFeatures()._get_embeddings(input_data)
    assert embeddings.shape == (41, 96)
    assert embeddings.max() == 38.666107
    assert embeddings.min() == -37.116394
    save_result(embeddings, "embeddings_from_empty_array.txt")


def test_audio_preproc():
    a = AudioFeatures()
    input_data = np.zeros(CHUNK).astype(np.int16)
    result = a(input_data)
    assert result
    assert a.feature_buffer.shape == (42, 96)
    assert a.melspectrogram_buffer.shape == (81, 32)
    save_result(a.feature_buffer, "audio_feature_buffer_from_empty_array.txt")
    save_result(a.melspectrogram_buffer, "audio_melspectogram_from_empty_array.txt")


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
    model.preprocessor.raw_data_buffer = deque([0] * (CHUNK * 10))
    total_size = filedata.size
    full_chunks = total_size // CHUNK
    np.arange(full_chunks * CHUNK, total_size, CHUNK)
    positive_predictions = []
    for i, chunk in iterate_chunks(filedata):
        prediction = model.predict(chunk)
        p = prediction['hey_jarvis_v0.1']
        if p > 0.9:
            print(f"{i} -> {prediction}")
            positive_predictions.append(prediction)

    assert 3 <= len(positive_predictions) <= 5
    save_predictions(positive_predictions, "predictions_hey_jarvis_from_empty_array.txt")


def test_detection_from_wav_negative():
    filedata = read_wav_int16('negative.wav')
    model = Model(["hey_jarvis_v0.1.onnx"])
    chunks = np.array_split(filedata, np.ceil(len(filedata) / CHUNK))
    positive_predictions = []
    for _, chunk in iterate_chunks(filedata):
        prediction = model.predict(chunk)
        p = prediction['hey_jarvis_v0.1']
        if p > 0.1:
            positive_predictions.append(prediction)

    assert len(positive_predictions) == 0


def read_wav_int16(wav_file) -> np.ndarray:
    with wave.open(wav_file, 'rb') as wf:
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return samples


def save_result(data: np.ndarray, filename):
    os.makedirs("test_results", exist_ok=True)
    np.savetxt(f'test_results/{filename}', data,
               delimiter=',', fmt='%f',
               header=f'Dim: {data.shape}')


def save_predictions(data: list[dict], filename):
    class FloatStringEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return str(obj)
            return super().default(obj)

    os.makedirs("test_results", exist_ok=True)
    with open(f'test_results/{filename}', "w") as file:
        json.dump(data, file, indent=4, cls=FloatStringEncoder)



def iterate_chunks(arr, chunk_size=CHUNK):
    # Calculate total size and number of full chunks
    total_size = arr.size
    n_chunks = total_size // chunk_size

    # Iterate over full chunks
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        yield i, arr[start_idx:end_idx]

    # Handle the last partial chunk if it exists
    if total_size % chunk_size != 0:
        start_idx = n_chunks * chunk_size
        yield n_chunks, arr[start_idx:]
