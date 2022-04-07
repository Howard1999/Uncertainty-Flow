import numpy as np
from numpy.random import uniform, randn


def load_wiggle(x_magnification=1.0, position_encoding=False, m=3):

    np.random.seed(0)
    Npoints = 300
    x = randn(Npoints) * 2.5 + 5  # uniform(0, 10, size=Npoints)

    def function(x):
        return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

    y = function(x)

    homo_noise_std = 0.25
    homo_noise = randn(*x.shape) * homo_noise_std
    y = y + homo_noise

    x = x[:, None]
    y = y[:, None]

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)

    X = (((x - x_means) / x_stds).astype(np.float32))
    Y = ((y - y_means) / y_stds).astype(np.float32)

    if position_encoding:
        x_p_list = [X]
        for i in range(m):
            x_p_list.append(np.sin((2**(i+1)) * X))
            x_p_list.append(np.cos((2**(i+1)) * X))
        X = np.concatenate(x_p_list, axis=1)

    return X[:, None, :] * x_magnification, Y[:, None, :]
