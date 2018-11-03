import bpy
import bmesh
from mathutils import Vector
import numpy as np

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import utils.blender_utils
import importlib
importlib.reload(utils.blender_utils)
from utils.blender_utils import init_greasy_pencil


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


def draw_cell(pos: tuple, gp_frame):
    x, y, z = pos
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = 500
    gp_stroke.points.add(count=2)
    gp_stroke.points[0].co = (x, 0, y)
    gp_stroke.points[1].co = (x + 0.5, 0, y)


def animate_automata(rule):
    automaton_size = 100
    automaton = Automaton_1D(automaton_size)
    nb_frames = 100
    bpy.context.scene.frame_end = nb_frames

    gp_layer = init_greasy_pencil()
    gp_frame = gp_layer.frames.new(0)
    #bpy.context.active_gpencil_brush.size = 100
    #bpy.context.active_gpencil_brush.strength = 1.
    # bpy.data.brushes["Draw Pencil"].size = 500

    for frame in range(1, nb_frames+1):
        #gp_frame = gp_layer.frames.new(frame)
        gp_frame = gp_layer.frames.copy(gp_frame)
        for i, cell in enumerate(automaton.space):
            if cell:
                draw_cell((i, frame, 0), gp_frame)
        automaton.update(rule)


rule_0 = {'111': 1, '110': 1, '101': 1, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
rule_sierpinski = {'111': 0, '110': 1, '101': 0, '100': 1, '011': 1, '010': 0, '001': 1, '000': 0}
rule_x = {'111': 0, '110': 0, '101': 0, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
animate_automata(rule_0)