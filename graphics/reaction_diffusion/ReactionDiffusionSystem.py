# adapted from https://github.com/benmaier/reaction-diffusion/blob/master/gray_scott.ipynb
# based on http://karlsims.com/rd.html

# proposed setup to ignore margins: https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/

import numpy as np
from typing import Tuple
from scipy import signal, ndimage

# Some sample tested configs

SYSTEM_CORAL_CONFIG = {
    'COEFF_A': 0.16,
    'COEFF_B': 0.08,
    'FEED_RATE': 0.060,
    'KILL_RATE': 0.062,
}

SYSTEM_BACTERIA_CONFIG = {
    'COEFF_A': 0.14,
    'COEFF_B': 0.06,
    'FEED_RATE': 0.035,
    'KILL_RATE': 0.066,
}

SYSTEM_SPIRALS_CONFIG = {
    'COEFF_A': 0.12,
    'COEFF_B': 0.08,
    'FEED_RATE': 0.020,
    'KILL_RATE': 0.050,
}

SYSTEM_ZEBRA_CONFIG = {
    'COEFF_A': 0.16,
    'COEFF_B': 0.08,
    'FEED_RATE': 0.035,
    'KILL_RATE': 0.060,
}


class ReactionDiffusionSystem:
    def __init__(self, shape: Tuple, config: dict, init_fun=None):
        """
        Instantiate a reaction-diffusion system with given config and initial grid state
        :param shape: size for each dimension of the system grid
        :param config: parameters that defined the behavior of the system
        :param init_fun: function that given a shape returns two initialized grids (one for each system component)
        """
        self.shape = shape
        self.config = config

        if init_fun is None:
            init_fun = get_init_state
        A, B = init_fun(self.shape)
        self.A = A
        self.B = B

    def update(self, delta_t=1.0):
        """
        Update system state
        :param delta_t: change in time for this iteration
        :return:
        """
        self.A, self.B = gray_scott_update(self.A, self.B,
                                           self.config['COEFF_A'], self.config['COEFF_B'],
                                           f=self.config['FEED_RATE'], k=self.config['KILL_RATE'],
                                           delta_t=delta_t)

    def run_simulation(self, steps: int, delta_t=1.0):
        """
        Update system state for the given number of steps
        :param steps: number of iterations to run
        :param delta_t: change in time for each iteration
        :return:
        """
        for step in range(steps):
            self.update(delta_t=delta_t)


def discrete_laplacian(M: np.ndarray, kernel: np.ndarray):
    """Get the discrete Laplacian of matrix M"""

    m_dims = len(M.shape)

    if m_dims == 2:
        axis = (0, 1)
        L = (
                (-2*m_dims) * M              # remove center 4x when 2D, 6 when 3D, etc.
                + np.roll(M, (0, -1), axis)  # right neighbor
                + np.roll(M, (0, +1), axis)  # left neighbor
                + np.roll(M, (-1, 0), axis)  # top neighbor
                + np.roll(M, (+1, 0), axis)  # bottom neighbor
             )
    elif m_dims == 3:
        axis = (0, 1, 2)
        L = (
                (-2*m_dims) * M                 # remove center 4x when 2D, 6 when 3D, etc.
                + np.roll(M, (0, -1, 0), axis)  # right neighbor
                + np.roll(M, (0, +1, 0), axis)  # left neighbor
                + np.roll(M, (-1, 0, 0), axis)  # top neighbor
                + np.roll(M, (+1, 0, 0), axis)  # bottom neighbor
                + np.roll(M, (0, 0, +1), axis)  # below neighbor
                + np.roll(M, (0, 0, -1), axis)  # above neighbor
             )
    else:
        raise NotImplementedError(f'discrete Laplacian not implemented for {m_dims} dimensions')

    return L


kernel_2d = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
kernel_2d_2 = np.array([
    [.05, .2, .05],
    [.2, -1, .2],
    [.05, .2, .05]
])
def discrete_laplacian_convolve(M: np.ndarray, kernel: np.ndarray):
    """Get the discrete Laplacian of matrix M via a 2D convolution operation
    Seems to perform way worse then the purely numpy implementation
    """
    return signal.convolve2d(M, kernel, mode='same', boundary='wrap')
    #return ndimage.filters.laplace(M, mode='wrap')


def gray_scott_update(A: np.ndarray, B: np.ndarray, coeff_A, coeff_B, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    :param A: concentration configuration A
    :param B: concentration configuration B
    :param coeff_A: diffusion coefficient for A
    :param coeff_B: diffusion coefficient for B
    :param f: feed rate (will be scaled by (1-A) so A doesn't exceed 1.0)
    :param k: kill rate
    :param delta_t: change in time for each iteration
    :return: new A and B values
    """

    # compute Laplacian of the two concentrations
    lA = discrete_laplacian_convolve(A, kernel_2d_2)
    lB = discrete_laplacian_convolve(B, kernel_2d_2)

    # apply the update formula
    AB_squared = A * B ** 2
    diff_A = (coeff_A * lA - AB_squared + f * (1 - A)) * delta_t
    diff_B = (coeff_B * lB + AB_squared - (k + f) * B) * delta_t

    A += diff_A
    B += diff_B

    return A, B


def get_init_state(shape, init_type='DEFAULT', random_influence=0.2):
    """
    Initialize a grid concentration state
    :param init_type: specify initialization mechanism
    :param shape: shape of the grid
    :param random_influence: describes how much noise is added (value between 1 and 0)
    :return: two initialized grids (one for each system component)
    """

    # start with a configuration where on every grid cell has a high concentration of chemical A
    A = (1 - random_influence) * np.ones(shape) + random_influence * np.random.random(shape)

    # assume there's only a bit of B everywhere
    B = random_influence * np.random.random(shape)

    if init_type == 'CENTER':
        # add a disturbance in the center
        center = np.array(shape) // 2
        r = np.array(shape) // 10

        left = center[0] - r[0]
        right = center[0] + r[0]
        bottom = center[1] - r[1]
        top = center[1] + r[1]

        m_dims = len(center)
        if m_dims == 2:
            A[left:right, bottom:top] = 0.50
            B[left:right, bottom:top] = 0.25
        elif m_dims == 3:
            below = center[2] - r[1]
            above = center[2] + r[1]
            A[left:right, bottom:top, below:above] = 0.50
            B[left:right, bottom:top, below:above] = 0.25
        else:
            raise NotImplementedError(f'discrete Laplacian not implemented for {m_dims} dimensions')

    return A, B
