import numpy as np

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


def test_audio():
    a = AudioFeatures()
    input_data = np.zeros(CHUNK).astype(np.int16)
    result = a(input_data)
    assert result
    assert a.feature_buffer.shape == (42, 96)
    assert a.melspectrogram_buffer.shape == (81,32)
