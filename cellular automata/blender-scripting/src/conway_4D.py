import bpy
import sys
import numpy as np

# Because Blender is stupid?
from __init__ import CONFIG_PATH

from abstract_GOL import AbstractGOL
import conway_3D
import gol_utils as utils


class ConwayGOL_4D(AbstractGOL):
    def __init__(self, N, config='GOL_4D_standard', seed=None):
        """
        4D Conway Game of Life
        :param N: 4D grid side size (resulting grid will be a NxNxNxN matrix)
        :param config: configuration for this GOL instance (cell survival and generation settings)
        """
        super().__init__(N, config, seed)
        self.grid = np.random.choice(2, (N,N,N,N))
    
    def update(self):
        """
        Update status of the grid
        """
        tmpGrid = self.grid.copy()
        for k in range(self.N):
            for z in range(self.N):
                for y in range(self.N):
                    for x in range(self.N):
                        neighbours = self.get_neighbours_count((k, z, y, x))
                        tmpGrid[k, z, y, x] = self.get_cell_newstate(self.grid[k, z, y, x], neighbours)
        self.grid = tmpGrid

    def get_neighbours_count(self, index):
        k, z, y, x = index
        neighbours_count = self.grid[max(0, k-1):min(k+2,self.N),
                                               max(0, z-1):min(z+2,self.N),
                                               max(0, y-1):min(y+2,self.N),
                                               max(0, x-1):min(x+2,self.N)].sum()
        neighbours_count -= self.grid[k, z, y, x]
        return neighbours_count


def main(_):
    # CONSTANTS
    num_frames_change = 2
    grid_side = 5
    obj_size = 0.7
    subdivisions = 10
    scale_factor=0.2

    #obj_generator = lambda x,y,z:icosphere_generator(obj_size, subdivisions, x, y, z)
    obj_generator = lambda x,y,z: utils.cube_generator(obj_size, x, y, z)
    obj_updater = lambda obj, grid, idx: utils.object_updater_hide(obj, grid[idx])

    utils.delete_all()
    gol = ConwayGOL_4D(grid_side)
    obj_grid = conway_3D.create_3Dgrid(gol, obj_generator)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x : conway_3D.frame_handler(x, obj_grid, gol,
                                                                   obj_updater,
                                                                   num_frames_change))


if __name__ == "__main__":
    main(sys.argv[1:])