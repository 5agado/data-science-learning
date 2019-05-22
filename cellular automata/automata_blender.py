import bpy
import bmesh
from mathutils import Vector
import numpy as np
import math
import itertools

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import Automaton
import automata_blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(automata_blender_utils)
from Automaton import *

from ds_utils.blender_utils import init_grease_pencil, draw_square, draw_circle, draw_cube, delete_all, render


def random_camera_pos(radius, azimuth, inclination, center=(0, 0, 0)):
    x = center[0] + radius * math.cos(azimuth) * math.sin(inclination)
    y = center[1] + radius * math.sin(azimuth) * math.sin(inclination)
    z = center[2] + radius * math.cos(inclination)

    camera = bpy.context.scene.objects['Camera']
    camera.location = (x, y, z)


def random_gp_material():
    line_color = np.random.rand(3)
    fill_color = np.random.rand(3)
    line_alpha, fill_alpha = [(1, 1), (0, 1)][np.random.randint(2)]  # random comb of alpha for line and fill

    if fill_color.sum() > 1.5:
        bpy.context.scene.world.color = (1, 1, 1)
    else:
        bpy.context.scene.world.color = (0, 0, 0)

    material = bpy.context.object.active_material.grease_pencil
    material.color = (line_color[0], line_color[1], line_color[2], line_alpha)
    material.fill_color = (fill_color[0], fill_color[1], fill_color[2], fill_alpha)


##################
# 1-D Automata
##################

def animate_1d_automata(rule, nb_frames=10, scale=1., material_index=0):
    # Init automata
    automaton_size = nb_frames*2
    automaton = Automaton1D(automaton_size, rule=rule)

    # Set middle cell as the only active one
    #automaton.grid = np.zeros(automaton_size, dtype=np.uint8)
    #automaton.grid[automaton_size // 2] = 1

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nb_frames

    gpencil_obj_name = "GPencil"
    gp_layer = init_grease_pencil(clear_layer=True, gpencil_obj_name=gpencil_obj_name)
    gp_frame = gp_layer.frames.new(0)

    gpencil = bpy.context.view_layer.objects[gpencil_obj_name]

    # center on middle cell
    cell_size = 1
    translate_vec = np.array([-(automaton_size/2), 0, 0])
    for frame in range(nb_frames):
        gp_frame = gp_layer.frames.copy(gp_frame)
        for i, cell in enumerate(automaton.grid):
            # maintain pyramid shape (render only if further from the center at least the current frame number)
            if cell and ((automaton_size // 2 - frame) <= i <= (automaton_size // 2 + frame)):
                # render cell
                centers = [
                    (i, frame, 0),  # normal center
                    #(i, frame - 1, automaton_size // 2 - frame),  # center down
                    #(i, frame - 1, -(automaton_size // 2) + frame),  # center up
                ]
                for center in centers:
                    centers_shifted = np.array(center) + translate_vec
                    draw_square(gp_frame, centers_shifted, cell_size, material_index=material_index)
                    #draw_cube(gp_frame, centers_shifted, cell_size, material_index=material_index)

        automaton.update()

    # scale automaton size along the growth axis
    if scale != 1.:
        gpencil.scale[0] = scale
        gpencil.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


rule_0 = {'111': 1, '110': 1, '101': 1, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
rule_sierpinski = {'111': 0, '110': 1, '101': 0, '100': 1, '011': 1, '010': 0, '001': 1, '000': 0}
rule_x = {'111': 0, '110': 0, '101': 0, '100': 1, '011': 1, '010': 1, '001': 1, '000': 0}
rule_106 = {'111': 0, '110': 1, '101': 1, '100': 0, '011': 1, '010': 0, '001': 1, '000': 0}

#animate_1d_automata(rule_sierpinski, nb_frames=50)


def explore_1d_automata(nb_frames, nb_runs: int, render_dir: Path):
    nb_simmetry = 4
    angle = 360/nb_simmetry
    scale = angle/90

    all_rules_config = list(itertools.product([0, 1], repeat=8))
    configs_idxs = np.random.choice(np.arange(len(all_rules_config)), nb_runs)
    for idx in configs_idxs:
        print(scale)
        print("#####")
        print(f"Rule {idx}")
        config = all_rules_config[idx]
        print(config)
        rule = dict(zip(['111', '110', '101', '100', '011', '010', '001', '000'], config))
        animate_1d_automata(rule, nb_frames=nb_frames, scale=scale, material_index=0)
        bpy.context.scene.frame_set(nb_frames)
        #random_camera_pos(np.random.randint(5, 200), np.random.randint(360), np.random.randint(360))
        random_gp_material()
        render(str(render_dir / f"rule_{idx}"), animation=False)
        render(str(render_dir / f"rule_{idx}"), animation=True)


#explore_1d_automata(50, nb_runs=20
#                    render_dir = Path.home() / "Downloads/automaton_1d/symm_4_colors_02")

##################
# 2-D Automata
##################

rule_gol = {'neighbours_count_born': 3,  # count required to make a cell alive
            'neighbours_maxcount_survive': 3,  # max number (inclusive) of neighbours that a cell can handle before dying
            'neighbours_mincount_survive': 2,  # min number (inclusive) of neighbours that a cell needs in order to stay alive
            }


# render automata with Grease Pencil
def animate_2d_automata(rule, nb_frames: 10, use_grease_pencil=True):
    nb_rows = 10
    nb_cols = 10
    gol = Automaton2D(nb_rows, nb_cols, rule, seed=11)

    FRAMES_SPACING = 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nb_frames * FRAMES_SPACING

    if use_grease_pencil:
        gp_layer = init_grease_pencil(clear_layer=True)
    else:
        obj_size = 0.7
        subdivisions = 2
        scale_factor = 0.2
        init_mat_color = (0.7, 0.1, 0.1)

        # obj_generator = lambda idx: automata_blender_utils.icosphere_generator(obj_size, subdivisions, idx[0], idx[1], 0)
        obj_generator = lambda idx: automata_blender_utils.cube_generator(obj_size, idx[0], idx[1], 0)

        obj_updater = lambda obj, grid, idx: automata_blender_utils.object_updater_hide(obj, grid[idx])
        # obj_updater = lambda obj, grid, idx: automata_blender_utils.object_updater_scale(obj, grid[idx],
        #                                                                                 scale_factor=scale_factor)
        # obj_updater = lambda obj, grid, idx: automata_blender_utils.object_updater_color_vector(
        # obj, grid[:, idx[0], idx[1]])

        delete_all()
        obj_grid = automata_blender_utils.create_grid(gol, obj_generator)
        # automata_blender_utils.init_materials(obj_grid, init_mat_color)

    gol.update()
    for frame in range(nb_frames):
        if use_grease_pencil:
            gp_frame = gp_layer.frames.new(frame * FRAMES_SPACING)
            for idx, val in np.ndenumerate(gol.grid):
                if val:
                    draw_square(gp_frame, (idx[0], idx[1], 0), 1)
        else:
            bpy.context.scene.frame_set(frame)
            automata_blender_utils.update_grid(obj_grid, gol, obj_updater)
        gol.update()


#animate_2d_automata(rule_gol, nb_frames=10, use_grease_pencil=True)


def animate_hexagonal_automata(p_freeze, p_melt, nb_frames: 10,
                               nb_rows: int, nb_cols: int,
                               material_index=0):
    automaton = HexagonalAutomaton(nb_rows=nb_rows, nb_cols=nb_cols, p_melt=p_melt, p_freeze=p_freeze)

    # Set middle cell as the only active one
    automaton.grid = np.zeros((nb_rows, nb_cols), dtype=np.uint8)
    automaton.grid[(nb_rows // 2, nb_cols//2)] = 1

    FRAMES_SPACING = 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nb_frames * FRAMES_SPACING

    gp_layer = init_grease_pencil(clear_layer=True)

    #gp_frame = gp_layer.frames.new(0)
    z = 0
    delete_all()
    for frame in range(0, nb_frames):
        if frame % 10 == 0:
            print("Frame {}".format(frame))
        #gp_frame = gp_layer.frames.copy(gp_frame)
        gp_frame = gp_layer.frames.new(frame * FRAMES_SPACING)

        # reduce reference size at each new frame
        size = 1/(frame+1)
        # Hexagonal shape size for grid adjustment
        hex_size = size*math.cos(math.pi/6)
        short_size = size/2
        #z += size/2
        z = 0
        for row in range(nb_rows):
            for col in range(nb_cols):
                if automaton.grid[row, col]:
                    # Calculate row and col position for the current cell
                    # taking into account hexagonal shape and shifting by growth
                    row_pos = (row - nb_rows//2) * (2*size - short_size)
                    col_pos = (col - nb_cols//2) * (2*hex_size) - hex_size
                    # shift even rows
                    if row % 2 == 0:
                        col_pos += hex_size

                    # Render cell
                    #automata_blender_utils.cube_generator(size, row_pos, col_pos, z)
                    #draw_cube(gp_frame, (row_pos, col_pos, z), size, material_index=material_index)
                    draw_circle(gp_frame, (row_pos, col_pos, z), size, 6, material_index=material_index)
        automaton.update()


p_freeze = [0, 1, 0., 0., 0, 0., 0., 1., 0, 0., 0., 0., 0., 0]
p_melt = [0, 0, 0., 0., 0., 0, 1, 0, 0., 1., 0, 1., 0., 0]
#animate_hexagonal_automata(p_freeze, p_melt, 10, nb_rows=120, nb_cols=120)


def explore_hexagonal_automata(nb_frames: int, nb_runs: int, nb_rows: int, nb_cols: int):
    render_dir = Path.home() / f"Downloads/automaton_hexagonal/flat_hexa_logo/{nb_frames}"
    render_dir.mkdir(exist_ok=True)
    with open(str(render_dir / "logs.txt"), 'w+') as f:
        for run in range(nb_runs):
            p_freeze = np.random.choice([1., 0.], 14)
            p_melt = np.random.choice([1., 0.], 14)
            print("#####")
            print(f"Run {run}")
            print(f"p_freeze {p_freeze}")
            print(f"p_melt {p_melt}")
            animate_hexagonal_automata(p_freeze, p_melt, nb_frames=nb_frames, nb_rows=nb_rows, nb_cols=nb_cols,
                                       material_index=0)
            bpy.context.scene.frame_set(nb_frames)
            #random_camera_pos(np.random.randint(5, 200), np.random.randint(360), np.random.randint(360))
            #andom_gp_material()

            render(str(render_dir / f"run_{run}"), animation=False)
            #render(str(render_dir / f"run_{run}"), animation=True)

            f.write(f"p_freeze:{p_freeze}-")
            f.write(f"p_melt:{p_melt}\n")


#for nb_frames in range(10, 20):
#    explore_hexagonal_automata(nb_frames, nb_runs=30, nb_rows=120, nb_cols=120)

# Dummy method to load some good configs from the exploratory generations
def load_good_configs(dir):
    imgs = dir.glob('*.png')
    good_runs = [int(img.stem.split('_')[1]) for img in imgs]
    confs = []
    with open(dir / 'logs.txt') as f:
        for i, line in enumerate(f):
            if i in good_runs:
                print(i)
                p_freeze, p_melt = line.split('-')
                p_freeze = list(map(float, p_freeze.split(':')[1][1:-2].split(' ')))
                p_melt = list(map(float, p_melt.split(':')[1][1:-2].split(' ')))
                confs.append((p_freeze, p_melt))
    print(len(confs))
    print(confs[0])
    return confs

load_good_configs(Path.home() / "Downloads/automaton_hexagonal/flat_hexa_logo/13")