import os
from os import path
import zipfile
import pickle
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib

import numpy as np

from .utils import *

# 2d data
def load_axis(base_dir='./dataset', position_encoding=False, m=3):

    if not path.exists(base_dir + '/gap_classification'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/gap_classification.zip',
                           filename=base_dir + '/gap_classification.zip')
        with zipfile.ZipFile(base_dir + '/gap_classification.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    file1 = base_dir + '/gap_classification/axis.pkl'

    with open(file1, 'rb') as f:
        axis_tupple = pickle.load(f)
        axis_x = axis_tupple[0].astype(np.float32)
        axis_y = axis_tupple[1].astype(np.float32)[:, np.newaxis]

        x_means, x_stds = axis_x.mean(axis=0), axis_x.std(axis=0)
        y_means, y_stds = axis_y.mean(axis=0), axis_y.std(axis=0)

        X = ((axis_x - x_means) / x_stds).astype(np.float32)
        Y = ((axis_y - y_means) / y_stds).astype(np.float32)
    
    if position_encoding:
        X = position_encode(X, m)

    return X[:, None, :], Y[:, None, :]