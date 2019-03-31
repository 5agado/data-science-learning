# Blender import system clutter
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import utils.blender_utils
import importlib
importlib.reload(utils.blender_utils)

from math import cos, sin, pi
import itertools

from utils.blender_utils import init_grease_pencil, draw_line, draw_cube

class LSystem(ABC):
    def __init__(self):
        self.variables = []
        self.constants = []
        self.axiom = []

    @abstractmethod
    def rules(self, val):
        pass

    @abstractmethod
    def rec_draw(self, draw_fun, vals: List[str], pos: Tuple[float], angle=0, depth=0, max_depth=3):
        pass

class DragonCurve(LSystem):

    def __init__(self):
        super().__init__()
        self.variables = ['X', 'Y']
        self.constants = ['-', '+', 'F']
        self.axiom = ['F', 'X']

    def rules(self, val):
        # verify that given var is in the system alphabet
        if val not in self.variables and val not in self.constants:
            raise Exception("{} not in the alphabet".format(val))
        if val in self.constants:
            return []
        elif val == 'X':
            return list('X+YF+')
        elif val == 'Y':
            return list('-FX-Y')

    def rec_draw(self, draw_fun, vals: List[str], pos: Tuple[float], angle=0, depth=0, max_depth=3):
        LINE_LENGTH = 1
        ANGLE_ADD = pi/2

        if depth >= max_depth:
            return angle, pos

        for val in vals:
            if val == '+':
                angle += ANGLE_ADD
            elif val == '-':
                angle -= ANGLE_ADD
            elif val == 'F':
                new_pos = (
                    pos[0] + LINE_LENGTH * cos(angle),
                    pos[1] + LINE_LENGTH * sin(angle),
                    pos[2]
                )
                draw_fun(pos, new_pos)
                pos = new_pos
            angle, pos = self.rec_draw(draw_fun, self.rules(val), pos, angle, depth=depth + 1, max_depth=max_depth)
        return angle, pos


class KochCurve(LSystem):

    def __init__(self):
        super().__init__()
        self.variables = ['F']
        self.constants = ['-', '+']
        self.axiom = ['F']

    def rules(self, val):
        # verify that given val is in the system alphabet
        if val not in self.variables and val not in self.constants:
            raise Exception("{} not in the alphabet".format(val))
        if val in self.constants:
            return []
        elif val == 'F':
            return list('F+F-F-F+F')

    def rec_draw(self, draw_fun, vals: List[str], pos: Tuple[float], angle=0, depth=0, max_depth=3):
        LINE_LENGTH = 1
        ANGLE_ADD = pi/2

        if depth >= max_depth:
            return angle, pos

        for val in vals:
            if val == '+':
                angle += ANGLE_ADD
            elif val == '-':
                angle -= ANGLE_ADD
            elif val == 'F':
                new_pos = (
                    pos[0] + LINE_LENGTH * cos(angle),
                    pos[1] + LINE_LENGTH * sin(angle),
                    pos[2]
                )
                draw_fun(pos, new_pos)
                pos = new_pos
            angle, pos = self.rec_draw(draw_fun, self.rules(val), pos, angle, depth=depth + 1, max_depth=max_depth)
        return angle, pos


class FractalPlant:

    def __init__(self):
        self.variables = ['X', 'F']
        self.constants = ['-', '+', '[', ']']
        self.axiom = ['X']

    def rules(self, val):
        # verify that given val is in the system alphabet
        if val not in self.variables and val not in self.constants:
            raise Exception("{} not in the alphabet".format(val))
        if val in self.constants:
            return [val]
        elif val == 'X':
            return list('F+[[X]-X]-F[-FX]+X')
        elif val == 'F':
            return ['F', 'F']

    def rec_draw(self, draw_fun, plant, pos: tuple, angle=0):
        LINE_LENGTH = 1

        ANGLE_ADD = 25
        skip = 0
        count = 0

        for i, val in enumerate(plant):
            # print(skip)
            count += 1
            if skip > 0:
                skip -= 1
                continue
            elif val not in self.variables and val not in self.constants:
                raise Exception("{} not in the alphabet".format(val))
            elif val in self.constants:
                if val == '+':
                    angle += ANGLE_ADD
                elif val == '-':
                    angle -= ANGLE_ADD
                elif val == '[':
                    skip = self.rec_draw(draw_fun, plant[i + 1:], (pos[0], pos[1], 0), angle)
                elif val == ']':
                    return count
            elif val == 'X':
                continue
            elif val == 'F':
                new_pos = (
                pos[0] + LINE_LENGTH * cos(angle * (pi / 180)), pos[1] + LINE_LENGTH * sin(angle * (pi / 180)), 0)
                draw_fun(pos, new_pos)
                # print(new_pos)
                pos = new_pos


def animate_lsystem(system: LSystem, max_depth: int, pos=(0, 0, 0), layer_name='GP_layer'):
    gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name=layer_name)
    gp_layer.frames.new(0)

    system.rec_draw(lambda x, y: draw(x, y, gp_layer), vals=system.axiom, pos=pos, max_depth=max_depth)


def animate_plant():
    fractal_plant = FractalPlant()

    NB_ITERATIONS = 5

    res = fractal_plant.axiom
    for i in range(1, NB_ITERATIONS):
        res = list(itertools.chain(*[fractal_plant.rules(x) for x in res]))

    gp_layer = init_grease_pencil(clear_layer=True)
    gp_layer.frames.new(0)

    fractal_plant.rec_draw(lambda x, y: draw(x, y, gp_layer), plant=res, pos=(0,0,0))


def draw(start: tuple, end: tuple, gp_layer):
    gp_frame = gp_layer.frames[-1]

    # Cube Transition
    # from scipy.spatial import distance
    # for i in range(1, 10):
    #     anim_frames = gp_layer.frames.copy(gp_frame)
    #     draw_cube(anim_frames, start, distance.euclidean(start, end)/i)

    # Rotating Line Transition
    # angle = 2 * pi / 10  # angle in radians
    # for i in range(1, 10):
    #     anim_frames = gp_layer.frames.copy(gp_frame)
    #     # Define stroke geometry
    #     radius = distance.euclidean(start, end)
    #     x = start[0] + radius * cos(angle * i)
    #     y = start[1] + radius * sin(angle * i)
    #     z = start[2]
    #     anim_end = (x, y, z)
    #     draw_line(anim_frames, start, anim_end)

    gp_frame = gp_layer.frames.copy(gp_frame)
    if gp_frame.frame_number%100 == 0:
        print("Writing to frame {}".format(gp_frame.frame_number))
    draw_line(gp_frame, start, end)


#animate_plant()
animate_lsystem(DragonCurve(), 11, layer_name='dragon_curve')
animate_lsystem(KochCurve(), 5, layer_name='koch_curve')
