import numpy as np

def load_ring_1d():
    np.random.seed(0)
    Npoints = 300

    def rotate(xy, theta):
        rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                  [np.sin(theta),  np.cos(theta)]])
        return rotate_matrix @ xy


    datapoint = []
    for _ in range(Npoints):
        xy = np.array([[0], [np.random.normal(2, 0.1)]])
        theta = np.random.uniform(-np.pi, np.pi)
        datapoint.append(rotate(xy, theta)[None, :])
    datapoint = np.concatenate(datapoint)
    return datapoint[:, 0, :], datapoint[:, 1, :]