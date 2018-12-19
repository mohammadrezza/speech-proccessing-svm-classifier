import scipy.io.wavfile as wav
import os
import progressbar
from blueprints import *

if __name__ == "__main__":

    files = os.listdir(DATA_SET_PATH)
    bar = progressbar.ProgressBar(maxval=len(files),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    print("Extacting features ...")


    def ext_feat():
        for i, file in enumerate(files):
            _, sig = wav.read(os.path.join(DATA_SET_PATH, file))
