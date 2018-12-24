import pickle
import os


def save(obj, fname, path):
    file = os.path.join(path, fname)
    with open(file, "wb") as f:
        f.write(pickle.dumps(obj))


def load(fname):
    with open(fname, "rb") as f:
        obj = pickle.loads(f.read())
    return obj
