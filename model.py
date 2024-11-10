import functools
import os
from collections import deque, defaultdict
from functools import partial
from typing import List, DefaultDict

import numpy as np
from utils import AudioFeatures

model_class_mappings = {  # TODO remove or replace by Hugo's values
    "timer": {
        "1": "1_minute_timer",
        "2": "5_minute_timer",
        "3": "10_minute_timer",
        "4": "20_minute_timer",
        "5": "30_minute_timer",
        "6": "1_hour_timer"
    }
}

# Define main model class
class Model:
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """

    def __init__(
            self,
            wakeword_models: List[str] = [],
            class_mapping_dicts: List[dict] = [],
            inference_framework: str = "tflite",
            **kwargs
    ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX/tflite models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        # Get model paths for pre-trained models if user doesn't provide models to load
        # pretrained_model_paths = openwakeword.get_pretrained_model_paths(inference_framework)
        wakeword_model_names = []

        for ndx, i in enumerate(wakeword_models):
            wakeword_model_names.append(os.path.splitext(os.path.basename(i))[0])

        # Create attributes to store models and metadata
        self.models = {}
        self.model_inputs = {}  # this contains model's .shape[1]
        self.model_outputs = {}
        self.model_prediction_function = {}  # and this
        self.class_mapping = {}

        # Support onnx framework only
        try:
            import onnxruntime as ort

            def onnx_predict(onnx_model, x):
                return onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})

        except ImportError:
            raise ValueError(
                "Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        print("ONNX runtime imported")

        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models

            # runtime session options
            sessionOptions = ort.SessionOptions()
            sessionOptions.inter_op_num_threads = 1
            sessionOptions.intra_op_num_threads = 1

            # is this the function from model
            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                         providers=["CPUExecutionProvider"])

            print(f"Session {self.models[mdl_name]}")
            self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
            self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
            pred_function = functools.partial(onnx_predict, self.models[mdl_name])
            self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            elif model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}
            print(f"Model {mdl_name} loaded")

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)

        print("Audio features done")

    def predict(self, x: np.ndarray):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected.
        """
        # Check input data type

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():  # always 1 for Hugo
            n_prepared_samples = self.preprocessor(x)
            if n_prepared_samples != 1280:
                print(f"N samples wrong: {n_prepared_samples}")

            ##  this
            prediction = self.model_prediction_function[mdl](
                # the function is "onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})"
                self.preprocessor.get_features(self.model_inputs[mdl])
            )

            predictions[mdl] = prediction[0][0][0]

            # Zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0

        # Update prediction buffer
        for mdl in predictions.keys():
            self.prediction_buffer[mdl].append(predictions[mdl])

        return predictions
