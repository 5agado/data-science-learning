from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod

from scipy import signal, ndimage


class AbstractAutomaton(ABC):
    def __init__(self, shape: Tuple, nb_states=2, seed=None):
        """
        Abstract automaton
        """
        if seed:
            np.random.seed(seed)

        self.shape = shape
        self.grid = np.random.choice(nb_states, shape)

    def update(self):
        """
        Update automaton grid
        """
        # with the given kernel is equivalent to count the neighbors (diagonal included)
        if len(self.grid.shape) == 2:
            neighbors_count = discrete_laplacian_convolve(self.grid, kernel_2d)
        else:
            neighbors_count = discrete_laplacian_nd_convolve(self.grid)

        # vectorize get-new-cell function and apply
        update_grid_func = np.vectorize(self.get_new_cell_state)
        self.grid = update_grid_func(self.grid, neighbors_count)

    def get_neighbors(self, index: Tuple[int]):
        pass

    @abstractmethod
    def get_new_cell_state(self, cell_current_state, neighbors):
        pass

kernel_2d = np.array([
    [1., 1., 1.],
    [1., 0., 1.],
    [1., 1., 1.]
])

def discrete_laplacian_convolve(M: np.ndarray, kernel: np.ndarray):
    """Get the discrete Laplacian of matrix M via a 2D convolution operation
    Seems to perform way worse then the purely numpy implementation
    """
    return signal.convolve2d(M, kernel, mode='same', boundary='wrap')


def discrete_laplacian_nd_convolve(M: np.ndarray):
    """Get the discrete Laplacian of matrix M
    """
    return ndimage.filters.laplace(M, mode='wrap')

class Automaton1D(AbstractAutomaton):
    def __init__(self, n: int, rule, states: int=2, seed=None):
        """
        1D Automaton
        :param n: number of cells
        """
        super().__init__((n, ), nb_states=states, seed=seed)

        self.n = n
        self.rule = rule

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

# Works for 2D and above
class AutomatonND(AbstractAutomaton):
    def __init__(self, shape: Tuple, config, seed=None):
        """
        """
        super().__init__(shape, seed=seed)

        self.config = config

    def get_new_cell_state(self, cell_current_state, neighbors_count):
        if neighbors_count == self.config['neighbours_count_born']:
            return 1
        if (neighbors_count < self.config['neighbours_mincount_survive']
                        or neighbors_count > self.config['neighbours_maxcount_survive']):
            return 0
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
        for state_idx, state in enumerate(self.states):
            if state in tmp_str:
                break

        if cell_current_state == 1:
            change_state_prob = self.p_melt[state_idx]
            return 0 if np.random.rand() <= change_state_prob else 1
        else:
            change_state_prob = self.p_freeze[state_idx]
            return 1 if np.random.rand() <= change_state_prob else 0


gol_rule = {'neighbours_count_born': 3,  # count required to make a cell alive
            'neighbours_maxcount_survive': 3,
            # max number (inclusive) of neighbours that a cell can handle before dying
            'neighbours_mincount_survive': 2,
            # min number (inclusive) of neighbours that a cell needs in order to stay alive
            }
