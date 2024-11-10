
import argparse
from datetime import datetime

import numpy as np
# Imports
import pyaudio

import model
import utils
from model import Model

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=utils.CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = model.Model(wakeword_models=[args.model_path])
else:
    print("Missing model name")
    exit(1)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("Listening for wakewords...")
    print(f"Models loaded {owwModel.models}")

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(utils.CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score_str = format(scores[-1], '.20f').replace("-", "")
            # print(scores)
            curr_score = scores[-1]
            # print(f"detect score {curr_score_str}")
            if curr_score > 0.3:
                current_time = datetime.now()
                print(current_time.strftime("%H:%M:%S"), curr_score)
        #     output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
        #     """
        #
        # # Print results table
        # print("\033[F"*(4*n_models+1))
        # print(output_string_header, "                             ", end='\r')
