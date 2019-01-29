import bpy
import bmesh
from mathutils import Vector
import numpy as np
import math

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata/blender-scripting/src"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import importlib

import utils.blender_utils
importlib.reload(utils.blender_utils)
from utils.blender_utils import init_grease_pencil, draw_square, draw_line

import conway_2D, gol_utils
importlib.reload(conway_2D)
importlib.reload(gol_utils)
CONFIG_PATH = str(SRC_PATH / '..' / 'GOL_config.ini')
#from conway_2D import CONFIG_PATH
from conway_2D import ConwayGOL_2D
from gol_utils import load_GOL_config


class Automaton_1D:
    def __init__(self, n: int, states: int = 2):
        """
        1D Automaton
        :param n: number of cells
        """
        self.n = n
        self.space = np.zeros(n, dtype=np.uint8)
        self.space[n // 2] = 1
        # np.array([0,0,0,0,1,0,0,0,0,0])#np.random.choice(2, n)

    def update(self, rule: dict):
        """
        Update automaton state
        """
        tmp_space = self.space.copy()
        for i in range(self.n):
            neighbours = self.get_neighbours(i)
            tmp_space[i] = rule["".join([str(s) for s in neighbours])]
        self.space = tmp_space

    def get_neighbours(self, i: int):
        if i == 0:
            return np.insert(self.space[:2], 0, self.space[-1])
        elif i == self.n - 1:
            return np.insert(self.space[-2:], 2, self.space[0])
        else:
            return self.space[max(0, i - 1):i + 2]


def animate_1d_automata(rule):
    automaton_size = 100
    automaton = Automaton_1D(automaton_size)

    NUM_FRAMES = 40
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES

    gp_layer = init_grease_pencil(clear_layer=True)
    gp_frame = gp_layer.frames.new(0)

    for frame in range(NUM_FRAMES):
        gp_frame = gp_layer.frames.copy(gp_frame)
        for i, cell in enumerate(automaton.space):
            if cell:
                p0 = (i/4, frame/4, 0)
                p1 = (p0[0], p0[1]+0.5, p0[2])
                draw_line(gp_frame, p0, p1)
        automaton.update(rule)


rule_0 = {'111': 1, '110': 1, '101': 1, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
rule_sierpinski = {'111': 0, '110': 1, '101': 0, '100': 1, '011': 1, '010': 0, '001': 1, '000': 0}
rule_x = {'111': 0, '110': 0, '101': 0, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
rule_106 = {'111': 0, '110': 1, '101': 1, '100': 0, '011': 1, '010': 0, '001': 1, '000': 0}
#animate_1d_automata(rule_106)


def animate_2d_automata():
    GOL_NB_ROWS = 50
    GOL_NB_COLS = 50
    gol = ConwayGOL_2D(GOL_NB_ROWS, GOL_NB_COLS,
                       load_GOL_config(CONFIG_PATH, 'GOL_2D_standard'),
                       seed=11)

    NUM_FRAMES = 50
    FRAMES_SPACING = 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES * FRAMES_SPACING

    gp_layer = init_grease_pencil(clear_layer=True)

    gol.update()
    for frame in range(NUM_FRAMES):
        gp_frame = gp_layer.frames.new(frame * FRAMES_SPACING)
        for i in range(gol.rows):
            for j in range(gol.cols):
                if gol.grid[i, j]:
                    draw_square(gp_frame, 1, (i, j, 0))
        gol.update()


animate_2d_automata()