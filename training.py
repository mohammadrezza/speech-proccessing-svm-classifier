import scipy.io.wavfile as wav
import os
import progressbar
from blueprints import *
from feature_extraction import extract
from utility import save

if __name__ == "__main__":

    files = os.listdir(DATA_SET_PATH)
    bar = progressbar.ProgressBar(maxval=len(files),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    print("Extacting features ...")
    features = []
    words_labels = []
    gender_lables = []


    def ext_feat():
        for i, file in enumerate(files):
            _, sig = wav.read(os.path.join(DATA_SET_PATH, file))
            feats = extract(sig)
            features.append(feats)
            words_labels.append(Label_Map[file[0]])
            gender_lables.append(Label_Map[file[1]])
            bar.update(i + 1)

    ext_feat()

    save(features, WORDS_FEATURES, MODELS_PATH)
    save(words_labels, WORDS_LABLES, MODELS_PATH)
    save(gender_lables, GENDER_LABLES, MODELS_PATH)
    bar.finish()
    print("Done.")

