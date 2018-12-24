from playsound import playsound
import pickle
import os
from blueprints import RESPONSE_SOUNDS_PATH, ResponseMap


def save(obj, fname, path):
    file = os.path.join(path, fname)
    with open(file, "wb") as f:
        f.write(pickle.dumps(obj))


def load(fname):
    with open(fname, "rb") as f:
        obj = pickle.loads(f.read())
    return obj


def play(audio_num):
    playsound(os.path.join(RESPONSE_SOUNDS_PATH, ResponseMap[audio_num]))
