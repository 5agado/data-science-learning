# Run cellular-automata in Blender
# input parameters need to be provided via animation-nodes

import bpy
import numpy as np
import math
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
from Automaton import HexagonalAutomaton

### Hexagonal CA
# nb_rows = 10
# nb_cols = 10
# seed = 11
# nb_epochs = 1

p_freeze = [0, 1, 0., 0., 0, 0., 0., 1., 0, 0., 0., 0., 0., 0]
p_melt = [0, 0, 0., 0., 0., 0, 1, 0, 0., 1., 0, 1., 0., 0]
# p_freeze = np.random.choice([1., 0.], 14)
# p_melt = np.random.choice([1., 0.], 14)

automaton = HexagonalAutomaton(nb_rows=nb_rows, nb_cols=nb_cols, p_melt=p_melt, p_freeze=p_freeze)
#automaton.update()

# run full sim for nb_epoch and store results
# load then the necessary layers based on current frame
if frame == 0 or frame == (nb_epochs+1):
    if init_grid is not None and len(init_grid) > 0:
        init_grid = np.array(init_grid).reshape((nb_rows, nb_cols, 4))
        automaton.grid = init_grid[:, :, 0] > 0.5
    else:
    # Set middle cell as the only active one
        automaton.grid = np.zeros((nb_rows, nb_cols), dtype=np.uint8)
        automaton.grid[(nb_rows // 2, nb_cols // 2)] = 1

    # reduce reference size at each new frame
    size = 1 / (frame + 1)
    # Hexagonal shape size for grid adjustment
    hex_size = size * math.cos(math.pi / 6)
    short_size = size / 2
    # z += size/2
    z = 0

    grid = []
    age = []
    for z in range(nb_epochs):
        if nb_epochs > 30 and z % 10 == 0:
            print("Frame {}".format(z))

        this_grid = []
        this_age = []
        alive = np.argwhere(automaton.grid == 1)
        growth_z = nb_epochs - z

        for (row, col) in alive:
            # Calculate row and col position for the current cell
            # taking into account hexagonal shape and shifting by growth
            row_pos = (row - nb_rows // 2) * (2 * size - short_size)
            col_pos = (col - nb_cols // 2) * (2 * hex_size) - hex_size
            # shift even rows
            if row % 2 == 0:
                col_pos += hex_size

            this_grid.append((row_pos, col_pos, growth_z))
            this_age.append(automaton.age[(row, col)])
        grid.append(this_grid)
        age.append(this_age)
        automaton.update()

    setattr(animation_nodes, "grid_hexa", grid)
    setattr(animation_nodes, "age_hexa", age)

grid = list(itertools.chain(*getattr(animation_nodes, "grid_hexa")[:frame]))
age = list(itertools.chain(*getattr(animation_nodes, "age_hexa")[:frame]))
