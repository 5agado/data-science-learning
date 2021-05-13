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

class ReactionDiffusionException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class ReactionDiffusionSystem:
    def __init__(self, shape: Tuple, config: dict, init_fun=None, validate_change_threshold=0.):
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

        self.validate_change_threshold = validate_change_threshold

    def update(self, delta_t=1.0):
        """
        Update system state
        :param delta_t: change in time for this iteration
        :return:
        """
        new_A, new_B = gray_scott_update(self.A, self.B,
                                           self.config['COEFF_A'], self.config['COEFF_B'],
                                           f=self.config['FEED_RATE'], k=self.config['KILL_RATE'],
                                           delta_t=delta_t)

        if self.validate_change_threshold > 0.:
            abs_change = abs((self.B - new_B).sum())
            if abs_change < self.validate_change_threshold:
                raise ReactionDiffusionException(f'abs_change = {abs_change}')

        self.A = new_A
        self.B = new_B


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
    [0, 1., 0],
    [1., -4., 1.],
    [0, 1., 0]
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
    #return ndimage.filters.convolve(M, kernel, mode='wrap')


def discrete_laplacian_nd_convolve(M: np.ndarray):
    """Get the discrete Laplacian of matrix M
    """
    #return discrete_laplacian(M, None)
    return ndimage.filters.laplace(M, mode='wrap')


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
    if len(A.shape) == 2:
        lA = discrete_laplacian_convolve(A, kernel_2d)
        lB = discrete_laplacian_convolve(B, kernel_2d)
    else:
        lA = discrete_laplacian_nd_convolve(A)
        lB = discrete_laplacian_nd_convolve(B)

    # apply the update formula
    AB_squared = A * B ** 2
    diff_A = (coeff_A * lA - AB_squared + f * (1. - A)) * delta_t
    diff_B = (coeff_B * lB + AB_squared - (k + f) * B) * delta_t

    return A + diff_A, B + diff_B


def get_init_state(shape, random_influence=0.2, mask: np.ndarray = None,
                   masked_A_val=0.5, masked_B_val=0.25):
    """
    Initialize a grid concentration state
    :param shape: shape of the grid
    :param random_influence: describes how much noise is added (value between 1 and 0)
    :param mask: optional mask to control concentration of chemicals
    :param masked_A_val: concentration of A when mask given
    :param masked_B_val: concentration of B when mask given
    :return: two initialized grids (one for each system component)
    """

    # start with a configuration where where chemical A is (1-random_influence) present on all grid,
    # add remaining noise scaled by random_influence
    A = (1 - random_influence) * np.ones(shape) + random_influence * np.random.random(shape)
    # chemical B is instead full noise scaled by random_influence
    B = random_influence * np.random.random(shape)

    # if a mask is given, edit A and B concentrations with specified values
    if mask is not None:
        assert mask.shape == shape, 'Mask shape does not match system shape'

        A[mask>0] = masked_A_val
        B[mask>0] = masked_B_val

    return A, B


def get_polygon_mask(shape: tuple, segments: int, radius: int, center=(0,0)):
    assert len(shape) == 2, 'Mask should be 2D'

    from PIL import Image, ImageDraw
    from math import pi, cos, sin

    # build polygon
    angle = 2 * pi / segments  # angle in radians
    polygon = []
    for i in range(segments):
        x = center[0] + radius * cos(angle * i)
        y = center[1] + radius * sin(angle * i)
        polygon.append((x, y))

    img = Image.new('L', (shape[0], shape[1]), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask


def get_cube_mask(shape: tuple, side: int, center=(0,0,0)):
    assert len(shape) == 3, 'Mask should be 3D'

    mask = np.zeros(shape, dtype=np.int)
    s = side//2
    mask[max(0, center[0] - s):center[0] + s,
         max(0, center[1] - s):center[1] + s,
         max(0, center[2] - s):center[2] + s] = 1

    return mask