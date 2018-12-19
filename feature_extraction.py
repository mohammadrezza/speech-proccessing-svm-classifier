from python_speech_features import sigproc, mfcc, delta
import numpy as np

SAMPLE_RATE = 11025
FRAME_LENGTH = int(SAMPLE_RATE * 0.025)
FRAME_STEP = FRAME_LENGTH - int(SAMPLE_RATE * 0.015)

def extract(sig):
    # framing
    sig_frames = sigproc.framesig(sig=sig, frame_len=FRAME_LENGTH, frame_step=FRAME_STEP)
    feat = []
