import numpy as np
from abc import ABC, abstractmethod


class AbstractGOL(ABC):
    def __init__(self, config, seed=None):
        """
        Abstract Conway Game of Life
        :param config: configuration for this GOL instance (cell survival and generation settings)
        """
        self.config = config
        if seed:
            np.random.seed(seed)

    @abstractmethod
    def update(self):
        """
        Update status of the grid
        """
        pass

    @abstractmethod
    def get_neighbours_count(self, index):
        pass

    def get_cell_newstate(self, cell_currstate, neighbours_count):
        if neighbours_count == self.config['neighbours_count_born']:
            return 1
        if (neighbours_count < self.config['neighbours_mincount_survive']
                        or neighbours_count > self.config['neighbours_maxcount_survive']):
            return 0
        return cell_currstate