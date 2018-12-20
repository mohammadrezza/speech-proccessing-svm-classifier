import pickle

def save(obj, fname):
    with open(fname, "wb") as f:
        f.write(pickle.dumps(obj))

def load(fname):
    with open(fname, "rb") as f:
        obj = pickle.loads(f.read())
    return obj
