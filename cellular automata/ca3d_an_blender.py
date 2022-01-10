# Run cellular-automata in Blender
# input parameters need to be provided via animation-nodes

import bpy
import numpy as np
import itertools

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata"
sys.path.append(str(SRC_PATH))

import Automaton
import automata_blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(automata_blender_utils)
from Automaton import AutomatonND

### 3D CA
# nb_rows = 10
# nb_cols = 10
# height = 10
# seed = 11
# nb_epochs = 1

ca3d_rule = {'neighbours_count_born': neighbours_count_born,  # count required to make a cell alive
            'neighbours_maxcount_survive': neighbours_maxcount_survive,
            # max number (inclusive) of neighbours that a cell can handle before dying
            'neighbours_mincount_survive': neighbours_mincount_survive,
            # min number (inclusive) of neighbours that a cell needs in order to stay alive
            }

grid_shape = (height, nb_rows, nb_cols)
gol = AutomatonND(grid_shape, ca3d_rule, seed=seed)

# run full sim for nb_epoch and store results
# load then the necessary layers based on current frame
if frame == 0 or frame == (nb_epochs+1):
    if init_grid is not None and len(init_grid) > 0:
        init_grid = np.array(init_grid).reshape((nb_rows, nb_cols, 4))
        gol.grid = np.zeros(grid_shape)
        gol.grid[height//2] = init_grid[:, :, 0] > 0.5
        gol.grid[:, height//2, :] = init_grid[:, :, 0] > 0.5
        gol.grid[:, :, height//2] = init_grid[:, :, 0] > 0.5

    grid = []
    age = []
    for z in range(nb_epochs):
        alive = np.argwhere(gol.grid == 1)
        grid.append([(g[1], g[2], g[0]) for g in alive])
        age.append([gol.age[g[0], g[1], g[2]] for g in alive])
        gol.update()

    setattr(animation_nodes, "grid_3d", grid)
    setattr(animation_nodes, "age_3d", age)

grid = list(getattr(animation_nodes, "grid_3d")[min(nb_epochs-1, frame)])
age = list(getattr(animation_nodes, "age_3d")[min(nb_epochs-1, frame)])