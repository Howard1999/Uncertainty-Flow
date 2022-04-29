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


def load_compose_1d():
    np.random.seed(0)

    def rotate(xy, theta):
        rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                  [np.sin(theta),  np.cos(theta)]])
        return rotate_matrix @ xy

    # part a.
    Npoints = 100
    data_a_x, data_a_y = np.random.normal(-1.5, 0.07, Npoints), np.random.normal(0.0, 0.07, Npoints)
    # part b.
    Npoints = 75
    data_b_xy = rotate(
                    np.array([np.random.normal(0, 0.02, Npoints), np.random.uniform(-0.4, 0.4, Npoints)]),
                    (- 45 / 180) * np.pi
                )
    data_b_x, data_b_y = data_b_xy[0, :] + -0.5, data_b_xy[1, :] - 0.1
    # part c.
    Npoints = 75
    data_c_xy = rotate(
                    np.array([np.random.normal(0, 0.01, Npoints), np.random.uniform(-0.3, 0.3, Npoints)]),
                    (45 / 180) * np.pi
                )
    data_c_x, data_c_y = data_c_xy[0, :] + 0.5, data_c_xy[1, :]
    # part d.
    Npoints = 50
    data_d_xy = rotate(
                    np.array([np.random.normal(0, 0.03, Npoints), np.random.uniform(-0.25, 0.25, Npoints)]),
                    0.
                )
    data_d_x, data_d_y = data_d_xy[0, :] + 1.5, data_d_xy[1, :] - 0.12
    
    data_x = np.concatenate([data_a_x, data_b_x, data_c_x, data_d_x]) * 1.8
    data_y = np.concatenate([data_a_y, data_b_y, data_c_y, data_d_y])

    return data_x[:, None], data_y[:, None]