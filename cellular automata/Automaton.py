from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod


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
        # use temporary grid as cell updates are applied simultaneously
        tmp_grid = self.grid.copy()

        # iterate over grid of arbitrary dimensions
        for idx, val in np.ndenumerate(self.grid):
            neighbors = self.get_neighbors(idx)
            tmp_grid[idx] = self.get_new_cell_state(self.grid[idx], neighbors)

        # replace automaton grid with new one
        self.grid = tmp_grid

    def get_neighbors(self, index: Tuple[int]):
        neighbors = self.grid
        # TODO condense in single operation
        for i, idx in enumerate(index):
            neighbors = neighbors.take(range(idx - 1, idx + 2), mode='wrap', axis=i)

        # TODO remove cell itself
        # can use np.delete, but this returns a new array, which might be computationally expensive

        return neighbors

    @abstractmethod
    def get_new_cell_state(self, cell_current_state, neighbors):
        pass


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


# TODO test full vectorization for update (use roll for 8 directions)
class Automaton2D(AbstractAutomaton):
    def __init__(self, nb_rows: int, nb_cols: int, config, seed=None):
        """
        """
        super().__init__((nb_rows, nb_cols), seed=seed)

        self.config = config

    def get_new_cell_state(self, cell_current_state, neighbors):
        neighbors_count = neighbors.sum()
        neighbors_count -= cell_current_state

        if neighbors_count == self.config['neighbours_count_born']:
            return 1
        if (neighbors_count < self.config['neighbours_mincount_survive']
                        or neighbors_count > self.config['neighbours_maxcount_survive']):
            return 0
        return cell_current_state


class AutomatonND(AbstractAutomaton):
    def __init__(self, shape: Tuple, config, seed=None):
        """
        """
        super().__init__((shape, config, seed))

        self.config = config

    def get_new_cell_state(self, cell_current_state, neighbors):
        neighbors_count = neighbors.sum()
        neighbors_count -= cell_current_state

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
