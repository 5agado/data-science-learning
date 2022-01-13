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
import mnca_utils
import automata_blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(mnca_utils)
importlib.reload(automata_blender_utils)
from Automaton import MultipleNeighborhoodAutomaton
from mnca_utils import get_circle_grid

### Multiple Neighborhood CA
# nb_rows = 10
# nb_cols = 10
# seed = 11
# nb_epochs = 1

configs = [
    {'neighbours_count_born': [0.300, 0.350],
     'neighbours_maxcount_survive': [0.350, 0.400],
     'neighbours_mincount_survive': [0.750, 0.850],
     },
]

kernels = [
    get_circle_grid(17, 17, radius_minmax=[2, 10]),

]


grid_shape = (nb_rows, nb_cols)
mnca = MultipleNeighborhoodAutomaton(grid_shape, configs=configs, kernels=kernels)
grid = get_circle_grid(mnca.shape[0], mnca.shape[1], radius_minmax=[0,50])
mnca.set_init_grid(grid)

# run full sim for nb_epoch and store results
# load then the necessary layers based on current frame
if frame == 0 or frame == (nb_epochs+1):
    if init_grid is not None and len(init_grid) > 0:
        init_grid = np.array(init_grid).reshape((nb_rows, nb_cols, 4))
        mnca.set_init_grid(init_grid[:, :, 0] > 0.5)

    grid = []
    age = []
    age_sum = mnca.age.copy()
    grid_age = []
    for z in range(nb_epochs):
        alive = np.argwhere(mnca.grid == 1)
        #alive = np.argwhere(mnca.grid == 2)
        grid.append([(g[0], g[1], z) for g in alive])
        #grid.append([(g[0], g[1], 0) for g in alive])
        age.append([mnca.age[g[0], g[1]] for g in alive])
        #age.append([age_sum[g[0], g[1]] for g in alive])
        mnca.update()
        age_sum += mnca.grid

    setattr(animation_nodes, "grid_mnca", grid)
    setattr(animation_nodes, "age_mnca", age)

grid = list(itertools.chain(*getattr(animation_nodes, "grid_mnca")[:frame]))
age = list(itertools.chain(*getattr(animation_nodes, "age_mnca")[:frame]))
# Age sum
# grid = getattr(animation_nodes, "grid_mnca")[frame]
# age = getattr(animation_nodes, "age_mnca")[frame]
