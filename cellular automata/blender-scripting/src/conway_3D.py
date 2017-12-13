import bpy
import sys
import numpy as np

# Because Blender is stupid?
from __init__ import CONFIG_PATH, SRC_PATH
sys.path.append(SRC_PATH)

import gol_utils as utils
from abstract_GOL import AbstractGOL
from __init__ import CONFIG_PATH

class ConwayGOL_3D(AbstractGOL):
    def __init__(self, N, config='GOL_3D_standard'):
        """
        3D Conway Game of Life
        :param N: 3D grid side size (resulting grid will be a NxNxN matrix)
        :param config: configuration for this GOL instance (cell survival and generation settings)
        """
        super().__init__(N, config)
        self.grid = np.random.choice(2, (N,N,N))

    def update(self):
        """
        Update status of the grid
        """
        tmpGrid = self.grid.copy()
        for z in range(self.N):
            for y in range(self.N):
                for x in range(self.N):
                    neighbours = self.get_neighbours_count((z, y, x))
                    tmpGrid[z, y, x] = self.get_cell_newstate(self.grid[z, y, x], neighbours)
        self.grid = tmpGrid

    def get_neighbours_count(self, index):
        z, y, x = index
        neighbours_count = self.grid[max(0, z-1):min(z+2,self.N),
                                           max(0, y-1):min(y+2,self.N),
                                           max(0, x-1):min(x+2,self.N)].sum()
        neighbours_count -= self.grid[z, y, x]
        return neighbours_count

# create grid of objects on current scene
# The object generator is responsible for the creation of a single object instance
def create_3Dgrid(gol, obj_generator):
    obj_grid = []
    for z in range(gol.N):
        plane = []
        for y in range(gol.N):
            row = []
            for x in range(gol.N):
                obj_generator(x, y, z)
                row.append(bpy.context.scene.objects.active)
            plane.append(row)
        obj_grid.append(plane)
    return obj_grid

# update grid of Blender objects to reflect gol status, then update gol.
def update_grid(obj_grid, gol, obj_updater):
    for z in range(gol.N):
        for y in range(gol.N):
            for x in range(gol.N):
                obj_updater(obj_grid[z][y][x], gol.grid, (z, y, x))
    gol.update()

# handler called at every frame change
def frame_handler(scene, grid, gol, obj_updater, num_frames_change):
    frame = scene.frame_current
    n = frame % num_frames_change
    if n == 0:
        update_grid(grid, gol, obj_updater)

def main(_):
    # CONSTANTS
    num_frames_change = 2
    grid_side = 5
    obj_size = 0.7
    subdivisions = 10
    scale_factor=0.2

    #obj_generator = lambda x,y,z:icosphere_generator(obj_size, subdivisions, x, y, z)
    obj_generator = lambda x,y,z: utils.cube_generator(obj_size, x, y, z)
    obj_updater = lambda obj,grid,idx:utils.object_updater_hide(obj, grid[idx])
    #obj_updater = lambda obj,grid,idx:utils.object_updater_scale(obj, grid[idx], scale_factor=scale_factor)

    utils.delete_all()
    gol = ConwayGOL_3D(grid_side, utils.load_GOL_config(CONFIG_PATH, 'GOL_3D_standard'))
    obj_grid = create_3Dgrid(gol, obj_generator)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x : frame_handler(x, obj_grid, gol,
                                                                   obj_updater,
                                                                   num_frames_change))

if __name__ == "__main__":
    main(sys.argv[1:])