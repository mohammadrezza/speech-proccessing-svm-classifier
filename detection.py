from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os
import numpy as np
from utility import load
from blueprints import *


def perform_pca(x, n):
    """
    if there are too many features versus few dataset, PCA is essential
    to achive better result.
    :param x: array-like, consist the features
    :param n: int, drop features into N features, it should be bigger than the size of the X
    :return: array-like, consist the features dropped to N
    """
    pca = PCA(n_components=n, whiten=True)
    pca.fit(np.array(x))
    pca_x = pca.transform(np.array(x))
    return pca_x


def classify(x, y):
    clf = SVC(C=1, gamma='auto', probability=True)
    clf.fit(np.array(x), np.array(y))
    return clf


if __name__ == "__main__":
    features = load(os.path.join(MODELS_PATH, WORDS_FEATURES))
    words_labels = load(os.path.join(MODELS_PATH, WORDS_LABLES))
    gender_labels = load(os.path.join(MODELS_PATH, GENDER_LABLES))

    pca_feats = perform_pca(features, 50)
    words_clf = classify(pca_feats, words_labels)
    gender_clf = classify(pca_feats, gender_labels)
