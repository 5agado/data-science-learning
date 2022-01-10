from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod
import itertools

from scipy import signal, ndimage
from scipy.fftpack import fft2, ifft2
#from cupyx.scipy import signal


class AbstractAutomaton(ABC):
    def __init__(self, shape: Tuple[int], nb_states=2, seed=None):
        """
        Abstract automaton
        """
        if seed is not None:
            np.random.seed(seed)

        self.shape = shape
        self.grid = np.random.choice(nb_states, shape)
        self.age = self.grid.copy()  # each cell age (number of epochs it stayed active)

    def set_init_grid(self, grid):
        assert grid.shape == self.grid.shape, 'Given grid does not match shape of automaton'
        self.grid = grid.copy().astype(int)
        self.age = self.grid.copy()

    @abstractmethod
    def update(self):
        """
        Update automaton grid
        """
        pass

    @abstractmethod
    def get_new_cell_state(self, cell_current_state, neighbors):
        pass


kernel_2d = np.array([
    [1., 1., 1.],
    [1., 0., 1.],
    [1., 1., 1.]
])


def get_kernel_2d_square(radius: int):
    size = (radius*2)+1
    return np.ones((size, size))


def discrete_laplacian_convolve(M: np.ndarray, kernel: np.ndarray):
    """Get the discrete Laplacian of matrix M via a 2D convolution operation
    Seems to perform way worse then the purely numpy implementation
    """
    return signal.convolve2d(M, kernel, mode='same', boundary='wrap')
    #return signal.fftconvolve(M, kernel, mode='same')
    #return fftconvolve2d(M, kernel)


def fftconvolve2d(x, y):
    # See https://stackoverflow.com/questions/46203604/need-a-circular-fft-convolution-in-python
    # This assumes y is "smaller" than x.
    f2 = ifft2(fft2(x, shape=x.shape) * fft2(y, shape=x.shape)).real
    f2 = np.roll(f2, (-((y.shape[0] - 1)//2), -((y.shape[1] - 1)//2)), axis=(0, 1))
    return f2


def discrete_laplacian_nd_convolve(M: np.ndarray):
    """Get the discrete Laplacian of matrix M
    """
    #return ndimage.filters.laplace(M, mode='wrap')  # default kernel used here is not good for our goals
    # can instead define a custom one like the following, but considers only the 6 orthogonal neighbors
    #from scipy.ndimage.filters import correlate1d, generic_laplace
    #def derivative2(input, axis, output, mode, cval):
    #    return correlate1d(input, [1, 0, 1], axis, output, mode, cval, 0)
    #return generic_laplace(M, derivative2, output=None, mode='wrap', cval=0.0)

    # the following shifts instead the full grid to account for all 26 neighbors
    all_3d_shifts = list(itertools.product([1, -1, 0], repeat=3))
    all_3d_shifts.remove((0, 0, 0))  # remove the no-shift combination
    axis = (0, 1, 2)
    L = np.sum([np.roll(M, shift, axis) for shift in all_3d_shifts], axis=0)
    return L


class Automaton1D(AbstractAutomaton):
    def __init__(self, n: int, rule, states: int=2, seed=None):
        """
        1D Automaton
        :param n: number of cells
        """
        super().__init__((n, ), nb_states=states, seed=seed)

        self.n = n
        self.rule = rule

    def update(self):
        # use temporary grid as cell updates are applied simultaneously
        tmp_grid = self.grid.copy()

        # iterate over vector
        for idx, val in np.ndenumerate(self.grid):
            neighbors = self.get_neighbors(idx)
            tmp_grid[idx] = self.get_new_cell_state(val, neighbors)

        # replace automaton grid with new one
        self.grid = tmp_grid

        # update age
        self.age = (self.age * self.grid) + self.grid

    def get_neighbors(self, index):
        index = index[0]
        if index == 0:
            return np.insert(self.grid[:2], 0, self.grid[-1])
        elif index == self.n - 1:
            return np.insert(self.grid[-2:], 2, self.grid[0])
        else:
            return self.grid[max(0, index - 1):index + 2]

    def get_new_cell_state(self, cell_current_state, neighbors):
        return self.rule["".join([str(s) for s in neighbors])]


# Conway's Game of Life rule for 2D CA
gol_rule = {'neighbours_count_born': 3,  # count required to make a cell alive
            'neighbours_maxcount_survive': 3,
            # max number (inclusive) of neighbours that a cell can handle before dying
            'neighbours_mincount_survive': 2,
            # min number (inclusive) of neighbours that a cell needs in order to stay alive
            }


class AutomatonND(AbstractAutomaton):
    def __init__(self, shape: Tuple, config, seed=None):
        """
        Handles both 2D and 3D automata. Works with higher dimensions too, but neighbors count wouldn't be appropriate.
        """
        super().__init__(shape, seed=seed)

        # allow for range of neighbours_count_born
        # if not a list, then convert to int_start and int_end (both inclusive)
        neighbours_count_born = config['neighbours_count_born']
        if not isinstance(neighbours_count_born, list):
            config['neighbours_count_born'] = [int(neighbours_count_born), int(neighbours_count_born)]
        self.config = config
        self.kernel = kernel_2d

    def update(self):
        if len(self.grid.shape) == 2:
            # with the given kernel is equivalent to count the neighbors (diagonal included)
            neighbors_count = discrete_laplacian_convolve(self.grid, self.kernel)
        else:
            neighbors_count = discrete_laplacian_nd_convolve(self.grid)

        # update the grid state
        self.grid = self.get_new_cell_state(self.grid.copy(), neighbors_count)

        # update age
        self.age = (self.age * self.grid) + self.grid

    def get_new_cell_state(self, cell_current_state, neighbors):
        born_mask = ((neighbors >= self.config['neighbours_count_born'][0]) &
                     (neighbors <= self.config['neighbours_count_born'][1]))
        cell_current_state[born_mask] = 1

        dead_mask = ((neighbors < self.config['neighbours_mincount_survive']) |
                     (neighbors > self.config['neighbours_maxcount_survive']))
        cell_current_state[dead_mask] = 0

        return cell_current_state


class HexagonalAutomaton(AbstractAutomaton):
    def __init__(self, nb_rows: int, nb_cols: int, p_melt, p_freeze, seed=None):
        """
        Hexagonal grid automaton
        Based on https://mathematica.stackexchange.com/a/39368/62536
        """
        super().__init__((nb_rows, nb_cols), seed=seed)

        self.states = np.array([
            '000000',
            '000001',
            '000011',
            '000101',
            '000111',
            '001001',
            '001011',
            '001101',
            '001111',
            '010101',
            '010111',
            '011011',
            '011111',
            '111111',
        ])

        self.p_melt = p_melt
        self.p_freeze = p_freeze

    def update(self):
        # use temporary grid as cell updates are applied simultaneously
        tmp_grid = self.grid.copy()

        # iterate over grid of arbitrary dimensions
        for idx, val in np.ndenumerate(self.grid):
            neighbors = self.get_neighbors(idx)
            tmp_grid[idx] = self.get_new_cell_state(self.grid[idx], neighbors)

        # replace automaton grid with new one
        self.grid = tmp_grid

        # update age
        self.age = (self.age * self.grid) + self.grid

    def get_neighbors(self, index):
        row, col = index
        neighbors = []

        # Row indexing fix for hexagonal setup based on row index
        if row % 2 != 0:
            to_col_left = -1
            to_col_right = +1
        else:
            to_col_left = 0
            to_col_right = 2

        # Wrapped selection of neighbors

        # "above" neighbors
        neighbors.extend(self.grid.take(row - 1, mode='wrap', axis=0)
                         .take(range(col + to_col_left, col + to_col_right), mode='wrap', axis=0).flatten())
        # sides neighbors
        neighbors.extend(self.grid.take(row, mode='wrap', axis=0)
                         .take(range(col - 1, col + 2), mode='wrap', axis=0).flatten())
        # "below" neighbors
        neighbors.extend(self.grid.take(row + 1, mode='wrap', axis=0)
                         .take(range(col + to_col_left, col + to_col_right), mode='wrap', axis=0).flatten())

        neighbors = np.array(neighbors, dtype=np.uint8)
        return neighbors

    def get_new_cell_state(self, cell_current_state, neighbors):
        # order neighbors as circular hexagonal cells
        neighbors = neighbors[np.array([0, 1, 4, 6, 5, 2])]

        # find the equivalent state. Enables rotation and reflection invariance
        # see https://stackoverflow.com/a/26924896/1677125
        tmp_str = "".join(map(str, neighbors))
        tmp_str += tmp_str
        state_idx = 0
        for state_idx, state in enumerate(self.states):
            if state in tmp_str:
                break

        if cell_current_state == 1:
            change_state_prob = self.p_melt[state_idx]
            return 0 if np.random.rand() <= change_state_prob else 1
        else:
            change_state_prob = self.p_freeze[state_idx]
            return 1 if np.random.rand() <= change_state_prob else 0


class MultipleNeighborhoodAutomaton(AbstractAutomaton):
    def __init__(self, shape: Tuple, configs: List, kernels: List, seed=None):
        """
        Multiple Neighborhood Cellular Automata (MNCA)
        See https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata
        """
        super().__init__(shape, seed=seed)

        assert len(configs) == len(kernels), 'Number of configs does not match number of kernels'
        self.kernels = kernels
        self.configs = configs

    def update(self):
        if len(self.grid.shape) == 2:
            neighborhood_avgs = []
            for kernel in self.kernels:
                neighborhood_avg = discrete_laplacian_convolve(self.grid, kernel) / kernel.sum()
                neighborhood_avgs.append(neighborhood_avg)
        else:
            raise NotImplementedError('Only 2D version is implemented')

        # update the grid state
        self.grid = self.get_new_cell_state(self.grid.copy(), neighborhood_avgs)

        # update age
        self.age = (self.age * self.grid) + self.grid

    def get_new_cell_state(self, cell_current_state, neighborhood_avgs):
        for i, config in enumerate(self.configs):
            neighborhood_avg = neighborhood_avgs[i]
            born_mask = ((neighborhood_avg >= config['neighbours_count_born'][0]) &
                         (neighborhood_avg <= config['neighbours_count_born'][1]))
            cell_current_state[born_mask] = 1
            dead_mask = ((neighborhood_avg >= config['neighbours_mincount_survive'][0]) &
                         (neighborhood_avg <= config['neighbours_mincount_survive'][1]))
            cell_current_state[dead_mask] = 0
            dead_mask = ((neighborhood_avg >= config['neighbours_maxcount_survive'][0]) &
                         (neighborhood_avg <= config['neighbours_maxcount_survive'][1]))
            cell_current_state[dead_mask] = 0
        return cell_current_state


# Examples for each of the supported CA can be found here
if __name__ == '__main__':
    nb_epochs = 10

    # 1D CA
    rule_sierpinski = {'111': 0, '110': 1, '101': 0, '100': 1, '011': 1, '010': 0, '001': 1, '000': 0}
    ca_1d = Automaton1D(100, rule=rule_sierpinski, seed=1)
    for epoch in range(nb_epochs):
        ca_1d.update()

    # 2D CA
    ca_2d = AutomatonND((100, 100), gol_rule, seed=1)
    for epoch in range(nb_epochs):
        ca_2d.update()

    # 3D CA
    ca_3d = AutomatonND((20, 20, 20), gol_rule, seed=1)
    for epoch in range(nb_epochs):
        ca_3d.update()

    # Hexagonal CA
    p_freeze = [0, 1, 0., 0., 0, 0., 0., 1., 0, 0., 0., 0., 0., 0]
    p_melt = [0, 0, 0., 0., 0., 0, 1, 0, 0., 1., 0, 1., 0., 0]
    ca_hexa = HexagonalAutomaton(nb_rows=20, nb_cols=20, p_melt=p_melt, p_freeze=p_freeze)
    for epoch in range(nb_epochs):
        ca_hexa.update()

    # Multiple Neighborhood CA
    configs = [
        {'neighbours_count_born': [0.190, 0.220],
         'neighbours_maxcount_survive': [0.350, 0.500],
         'neighbours_mincount_survive': [0.750, 0.850],
         },
        {'neighbours_count_born': [0.410, 0.550],
         'neighbours_maxcount_survive': [0.100, 0.280],
         'neighbours_mincount_survive': [0.120, 0.150],
         },
    ]

    kernels = [
        np.ones([9, 9]),
        np.ones([10, 10]),

    ]
    mnca = MultipleNeighborhoodAutomaton((100, 100), configs=configs, kernels=kernels)
    for epoch in range(nb_epochs):
        mnca.update()
