
# Imports
import os
from collections import deque
from multiprocessing.pool import ThreadPool
from typing import Union, List, Callable, Deque

import numpy as np

CHUNK = 1280

# Base class for computing audio features using Google's speech_embedding
# model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures():
    """
    A class for creating audio features from audio data, including melspectograms and Google's
    `speech_embedding` features.
    """
    def __init__(self,
                 sr: int = 16000,
                 ncpu: int = 1
                 ):
        """
        Initialize the AudioFeatures object.

        Args:
            melspec_model_path (str): The path to the model for computing melspectograms from audio data
            embedding_model_path (str): The path to the model for Google's `speech_embedding` model
            sr (int): The sample rate of the audio (default: 16000 khz)
            ncpu (int): The number of CPUs to use when computing melspectrograms and audio features (default: 1)
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            device (str): The device to use when running the models, either "cpu" or "gpu" (default is "cpu".)
                          Note that depending on the inference framework selected and system configuration,
                          this setting may not have an effect. For example, to use a GPU with the ONNX
                          framework the appropriate onnxruntime package must be installed.
        """
        # Initialize the models with the appropriate framework
        try:
            import onnxruntime as ort
        except ImportError:
            raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        melspec_model_path = os.path.join("models", "melspectrogram.onnx")
        embedding_model_path = os.path.join("models", "embedding_model.onnx")

        print("embedding models loaded")

        # Initialize ONNX options
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = ncpu
        sessionOptions.intra_op_num_threads = ncpu

        # Melspectrogram model
        self.melspec_model = ort.InferenceSession(melspec_model_path, sess_options=sessionOptions,
                                                  providers=["CPUExecutionProvider"])
        self.onnx_execution_provider = self.melspec_model.get_providers()[0]
        self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})

        # Audio embedding model
        self.embedding_model = ort.InferenceSession(embedding_model_path, sess_options=sessionOptions,
                                                    providers=["CPUExecutionProvider"])
        self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()

        # Create databuffers with empty/random data
        self.raw_data_buffer: Deque = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
        self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
        self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
        self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history
        print("audio features init done")

    def reset(self):
        """Reset the internal buffers"""
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))

    def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
        """
        Function to compute the mel-spectrogram of the provided audio samples.

        Args:
            x (Union[np.ndarray, List]): The input audio data to compute the melspectrogram from
            melspec_transform (Callable): A function to transform the computed melspectrogram. Defaults to a transform
                                          that makes the ONNX melspectrogram model closer to the native Tensorflow
                                          implementation from Google (https://tfhub.dev/google/speech_embedding/1).

        Return:
            np.ndarray: The computed melspectrogram of the input audio data
        """
        # Get input data and adjust type/shape as needed
        x = np.array(x).astype(np.int16) if isinstance(x, list) else x
        if x.dtype != np.int16:
            raise ValueError("Input data must be 16-bit integers (i.e., 16-bit PCM audio)."
                             f"You provided {x.dtype} data.")
        x = x[None, ] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype != np.float32 else x

        # Get melspectrogram
        outputs = self.melspec_model_predict(x)
        spec = np.squeeze(outputs[0])

        # Arbitrary transform of melspectrogram
        spec = melspec_transform(spec)

        return spec

    def _get_embeddings_from_melspec(self, melspec):
        """
        Computes the Google `speech_embedding` features from a melspectrogram input

        Args:
            melspec (np.ndarray): The input melspectrogram

        Returns:
            np.ndarray: The computed audio features/embeddings
        """
        if melspec.shape[0] != 1:
            melspec = melspec[None, ]
        embedding = self.embedding_model_predict(melspec)
        return embedding

    def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8, **kwargs):
        """Function to compute the embeddings of the provide audio samples."""
        spec = self._get_melspectrogram(x, **kwargs)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:  # truncate short windows
                windows.append(window)

        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model_predict(batch)
        return embedding

    def get_embedding_shape(self, audio_length: float, sr: int = 16000):
        """Function that determines the size of the output embedding array for a given audio clip length (in seconds)"""
        x = (np.random.uniform(-1, 1, int(audio_length*sr))*32767).astype(np.int16)
        return self._get_embeddings(x).shape


    def _streaming_melspectrogram(self, n_samples):
        """Note! There seem to be some slight numerical issues depending on the underlying audio data
        such that the streaming method is not exactly the same as when the melspectrogram of the entire
        clip is calculated. It's unclear if this difference is significant and will impact model performance.
        In particular padding with 0 or very small values seems to demonstrate the differences well.
        """
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples-160*3:]))
        )

        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _buffer_raw_data(self, x):
        """
        Adds raw audio data to the input buffer
        """
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_features(self, x):
        # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
        processed_samples = 0

        if self.raw_data_remainder.shape[0] != 0:
            print('raw_data_remainder.shape[0] != 0')
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)

        if self.accumulated_samples + x.shape[0] >= CHUNK:
            # print(f'samoles >= CHUNK; {self.accumulated_samples.bit_length()} - {x.shape[0]}')
            remainder = (self.accumulated_samples + x.shape[0]) % CHUNK
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            print('sample < CHUNK')
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        # Only calculate melspectrogram once minimum samples are accumulated
        if self.accumulated_samples >= CHUNK and self.accumulated_samples % CHUNK == 0:
            # print(f'Samples {self.accumulated_samples}')
            self._streaming_melspectrogram(self.accumulated_samples)

            # Calculate new audio embeddings/features based on update melspectrograms
            for i in np.arange(self.accumulated_samples//CHUNK-1, -1, -1):
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    self.feature_buffer = np.vstack((self.feature_buffer,
                                                    self.embedding_model_predict(x)))

            # Reset raw data buffer counter
            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

        return processed_samples if processed_samples != 0 else self.accumulated_samples

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)

    def __call__(self, x):
        # used in inference
        return self._streaming_features(x)

