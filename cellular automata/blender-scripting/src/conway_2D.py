import bpy
import sys
import numpy as np

# Because Blender is stupid?
from __init__ import CONFIG_PATH

import gol_utils as utils
from conway_3D import ConwayGOL_3D
from abstract_GOL import AbstractGOL


class ConwayGOL_2D(AbstractGOL):
    def __init__(self, rows: int, cols: int, config='GOL_2D_standard', seed=None):
        """
        2D Conway Game of Life
        :param rows: grid number of rows
        :param cols: grid number of cols
        :param config: config: configuration for this GOL instance (cell survival and generation settings)
        """
        super().__init__(config, seed)
        self.rows = rows
        self.cols = cols
        self.grid = np.random.choice(2, (rows, cols))
    
    def update(self):
        """
        Update status of the grid
        """
        tmpGrid = self.grid.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                neighbours = self.get_neighbours_count((i, j))
                tmpGrid[i, j] = self.get_cell_newstate(self.grid[i, j], neighbours)
        self.grid = tmpGrid

    def get_neighbours_count(self, index):
        i, j = index
        neighbours_count = self.grid.take(range(i-1, i+2), mode='wrap', axis=0)\
                                    .take(range(j-1, j+2), mode='wrap', axis=1).sum()
        #neighbours_count = self.grid[max(0, i-1):min(i+2, self.rows), max(0, j-1):min(j+2, self.cols)].sum()
        neighbours_count -= self.grid[i, j]
        return neighbours_count

#######################################
#            UTIL METHODS             #
#######################################


# create grid of objects on current scene
# The object generator is responsible for the creation of a single object instance
def create_grid(gol, obj_generator):
    obj_grid = []
    for i in range(gol.rows):
        row = []
        for j in range(gol.cols):
            obj_generator(i, j, 0)
            row.append(bpy.context.view_layer.objects.active)
        obj_grid.append(row)
    return obj_grid


# update grid of Blender objects to reflect gol status, then update gol.
def update_grid(obj_grid, gol, obj_updater):
    for i in range(gol.rows):
        for j in range(gol.cols):
            obj_updater(obj_grid[i][j], gol.grid, (i, j))
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
    init_mat_color = (0.7, 0.1, 0.1)

    #obj_generator = lambda x,y,z:utils.icosphere_generator(obj_size, subdivisions, x, y, z)
    obj_generator = lambda x,y,z:utils.cube_generator(obj_size, x, y, z)
    #obj_updater = lambda obj,grid,idx:utils.object_updater_hide(obj, grid[idx])
    #obj_updater = lambda obj,grid,idx:utils.object_updater_scale(obj, grid[idx], scale_factor=scale_factor)
    obj_updater = lambda obj,grid,idx:utils.object_updater_color_vector(obj, grid[:, idx[0], idx[1]]) #check order

    utils.delete_all()
    gol = ConwayGOL_3D(grid_side,
                       utils.load_GOL_config(CONFIG_PATH, 'GOL_3D_standard'),
                       seed=42)
    obj_grid = create_grid(gol, obj_generator)
    utils.init_materials(obj_grid, init_mat_color)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x : frame_handler(x, obj_grid, gol,
                                                                   obj_updater,
                                                                   num_frames_change))


if __name__ == "__main__":
    main(sys.argv[1:])