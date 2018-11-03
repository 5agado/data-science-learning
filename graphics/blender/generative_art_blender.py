# Blender import system clutter
import sys
import bpy
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import utils.blender_utils
import importlib
importlib.reload(utils.blender_utils)

from math import cos, sin, pi
import itertools

from utils.blender_utils import init_greasy_pencil

class DragonCurve:

    def __init__(self):
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

    def rec_draw(self, draw_fun, vals, pos: tuple, angle=0, depth=0, max_depth=3):
        LINE_LENGTH = 1
        ANGLE_ADD = 90

        if depth >= max_depth:
            return angle, pos

        for val in vals:
            if val == '+':
                angle += ANGLE_ADD
            elif val == '-':
                angle -= ANGLE_ADD
            elif val == 'F':
                new_pos = (
                pos[0] + LINE_LENGTH * cos(angle * (pi / 180)), pos[1] + LINE_LENGTH * sin(angle * (pi / 180)), 0)
                draw_fun(pos, new_pos)
                pos = new_pos
            angle, pos = self.rec_draw(draw_fun, self.rules(val), pos, angle, depth=depth + 1, max_depth=max_depth)
        return angle, pos


class KochCurve:

    def __init__(self):
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

    def rec_draw(self, draw_fun, vals, pos: tuple, angle=0, depth=0, max_depth=3):
        LINE_LENGTH = 1
        ANGLE_ADD = 90

        if depth >= max_depth:
            return angle, pos

        for val in vals:
            if val == '+':
                angle += ANGLE_ADD
            elif val == '-':
                angle -= ANGLE_ADD
            elif val == 'F':
                new_pos = (
                pos[0] + LINE_LENGTH * cos(angle * (pi / 180)), pos[1] + LINE_LENGTH * sin(angle * (pi / 180)), 0)
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


def animate_plant():
    fractal_plant = FractalPlant()

    NB_ITERATIONS = 6

    res = fractal_plant.axiom
    for i in range(1, NB_ITERATIONS):
        res = list(itertools.chain(*[fractal_plant.rules(x) for x in res]))

    gp_layer = init_greasy_pencil()
    gp_layer.frames.new(0)

    fractal_plant.rec_draw(lambda x, y: draw_line(x, y, gp_layer), plant=res, pos=(0,0,0))


def animate_dragon_curve():
    l_system = DragonCurve()
    gp_layer = init_greasy_pencil()
    gp_layer.frames.new(0)

    l_system.rec_draw(lambda x, y: draw_line(x, y, gp_layer), vals=l_system.axiom, pos=(0,0,0), max_depth=11)


def animate_koch_curve():
    l_system = KochCurve()
    gp_layer = init_greasy_pencil()
    gp_layer.frames.new(0)

    l_system.rec_draw(lambda x, y: draw_line(x, y, gp_layer), vals=l_system.axiom, pos=(0,0,0), max_depth=5)


def draw_line(start: tuple, end: tuple, gp_layer):
    gp_frame = gp_layer.frames.copy(gp_layer.frames[-1])
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = 20
    gp_stroke.points.add(count=2)
    gp_stroke.points[0].co = start
    gp_stroke.points[1].co = end


animate_plant()
animate_dragon_curve()
animate_koch_curve()