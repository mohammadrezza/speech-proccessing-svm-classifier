from python_speech_features import sigproc, mfcc, delta
import numpy as np

SAMPLE_RATE = 11025
FRAME_LENGTH = int(SAMPLE_RATE * 0.025)
FRAME_STEP = FRAME_LENGTH - int(SAMPLE_RATE * 0.015)
PRE_EMPH = 0.97
WINDOW_LENGTH = 0.025
WINDOW_STEP = 0.010
WINDOW_FUNCTION = np.hamming


def extract(sig):
    # framing
    sig_frames = sigproc.framesig(sig=sig, frame_len=FRAME_LENGTH, frame_step=FRAME_STEP)
    feat = []

    def calc_all_feat(feat_coeffs):
        feat.extend(feat_coeffs.max(axis=0))
        feat.extend(feat_coeffs.min(axis=0))
        feat.extend(feat_coeffs.mean(axis=0))
        feat.extend(feat_coeffs.var(axis=0))

    # region calculate mfcc features
    mfcc_feat = mfcc(signal=sig_frames, samplerate=SAMPLE_RATE, winlen=WINDOW_LENGTH, winstep=WINDOW_STEP,
                     numcep=13, preemph=PRE_EMPH, winfunc=WINDOW_FUNCTION)
    mfcc_feat_delta = delta(mfcc_feat, 20)
    mfcc_feat_delta_delta = delta(mfcc_feat_delta, 20)

    calc_all_feat(mfcc_feat)
    calc_all_feat(mfcc_feat_delta)
    calc_all_feat(mfcc_feat_delta_delta)

    # endregion
