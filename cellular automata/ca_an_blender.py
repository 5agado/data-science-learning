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
from Automaton import AutomatonND, gol_rule
from automata_blender_utils import get_init_grid_from_image

### 2D CA
# nb_rows = 10
# nb_cols = 10
# seed = 11
# nb_epochs = 1
grid_shape = (nb_rows, nb_cols)
gol = AutomatonND((nb_rows, nb_cols), gol_rule, seed=seed)

# run full sim for nb_epoch and store results
# load then the necessary layers based on current frame
if frame == 0 or frame == (nb_epochs+1):
    if init_grid_image is not None and init_grid_image:
        init_grid = get_init_grid_from_image(init_grid_image, shape=grid_shape, threshold=0.5, invert=False)
        gol.grid = init_grid
    elif init_grid is not None and len(init_grid) > 0:
        init_grid = np.array(init_grid).reshape((nb_rows, nb_cols, 4))
        gol.grid = init_grid[:, :, 0] > 0.5

    grid = []
    age = []
    for z in range(nb_epochs):
        alive = np.argwhere(gol.grid == 1)
        grid.append([(g[0], g[1], z) for g in alive])
        age.append([gol.age[g[0], g[1]] for g in alive])
        gol.update()

    setattr(animation_nodes, "grid", grid)
    setattr(animation_nodes, "age", age)

grid = list(itertools.chain(*getattr(animation_nodes, "grid")[:frame]))
age = list(itertools.chain(*getattr(animation_nodes, "age")[:frame]))
