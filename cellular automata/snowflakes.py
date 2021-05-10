import bpy
import bmesh

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata"
GROWTH_PATH = UTILS_PATH / "graphics/blender"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))
sys.path.append(str(GROWTH_PATH))

import Automaton
import growth_anim
import automata_blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(automata_blender_utils)
importlib.reload(growth_anim)
from Automaton import *

from ds_utils.blender_utils import delete_all, create_object, render
from growth_anim import anim_particles, anim_objs
from automata_blender_utils import calculate_hexagonal_cell_position


def randomize_bsdf():
    bsdf = bpy.data.materials[1].node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Specular'].default_value = np.random.rand()
    bsdf.inputs['Metallic'].default_value = np.random.rand()
    bsdf.inputs['Roughness'].default_value = np.random.rand()
    bsdf.inputs['Base Color'].default_value = np.random.rand(4)
    bsdf.inputs['Base Color'].default_value[3] = 1.


def single_run(automaton, nb_frames: int, render_path=None):
    nb_rows = automaton.shape[0]
    nb_cols = automaton.shape[1]

    cell_size = 0.6
    objs_verts = []
    for frame in range(0, nb_frames):
        if frame % 10 == 0:
            print("Automaton update - frame {}".format(frame))
        automaton.update()

        vertices = []
        negatives = []
        z = -frame
        for row in range(nb_rows):
            for col in range(nb_cols):
                x, y = calculate_hexagonal_cell_position(row, col, nb_rows, nb_cols, cell_size)

                if automaton.grid[row, col]:
                    vertices.append((x, y, z))
                else:
                    negatives.append((x, y, z))

        vertices = np.array(vertices)
        objs_verts.append(vertices)

    #anim_objs(objs_verts)
    #return

    obj_name = 'snowflake'
    if obj_name not in bpy.context.scene.objects:
        create_object(vertices, edges=[], faces=[], obj_name=obj_name)
    else:
        obj = bpy.context.scene.objects[obj_name]
        mesh = obj.data
        bm = bmesh.new()

        # convert the current mesh to a bmesh (must be in edit mode)
        bpy.ops.object.mode_set(mode='EDIT')
        bm.from_mesh(mesh)
        bpy.ops.object.mode_set(mode='OBJECT')  # return to object mode

        for v in bm.verts:
            bm.verts.remove(v)
        for v in vertices:
            bm.verts.new(v)

        # make the bmesh the object's mesh
        bm.to_mesh(mesh)
        bm.free()  # always do this when finished

    if render_path:
        randomize_bsdf()
        render(render_path, animation=False)


def growth(automaton):
    nb_frames = 10

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nb_frames

    nb_rows = automaton.shape[0]
    nb_cols = automaton.shape[1]

    types = np.random.choice(['ELLIPSOID', 'CUBE', 'BALL'], nb_rows+nb_cols)

    # Main metaball
    bpy.ops.object.metaball_add(type='BALL', enter_editmode=True, location=(0, 0, 0))
    mball = bpy.context.active_object.data

    # build grid
    z = 0
    init_size = 0.1
    size = 0.5
    for row in range(nb_rows):
        for col in range(nb_cols):
            row_dist = abs(automaton_center[0] - row)
            col_dist = abs(automaton_center[1] - col)

            x, y = calculate_hexagonal_cell_position(row, col, nb_rows, nb_cols, size)

            # Add metaball element
            mball.elements.new()
            element = mball.elements[-1]
            element.co = (x, y, z)
            element.hide = True
            element.keyframe_insert("hide")
            element.type = types[row_dist+col_dist]
            element.radius = size
            if element.type == 'CUBE':
                element.radius = size
                element.size_x = size/2
                element.size_y = size/2
                element.size_z = 0.01

    for frame in range(0, nb_frames):
        if frame % 10 == 0:
            print("Frame {}".format(frame))

        automaton.update()
        bpy.context.scene.frame_set(frame)

        for row in range(nb_rows):
            for col in range(nb_cols):
                element = mball.elements[row*nb_cols + col]
                #element.type = 'CUBE'
                #element.radius = size
                if automaton.grid[row, col]:
                    element.hide = False
                    element.keyframe_insert("hide")
                else:
                    element.hide = True
                    element.keyframe_insert("hide")

def main(nb_rows, nb_cols):
    print("-------------------------------------")

    p_freeze = [0, 1, 0., 0., 0, 0., 0., 1., 0, 0., 0., 0., 0., 0]
    p_melt = [0, 0, 0., 0., 0., 0, 1, 0, 0., 1., 0, 1., 0., 0]
    # p_freeze = np.random.choice([1., 0.], 14)
    # p_melt = np.random.choice([1., 0.], 14)
    configs = automata_blender_utils.load_good_configs(Path.home() / "Documents/graphics/generative_art_output/snowflakes/flat_hexa_logo/19")
    p_freeze, p_melt = configs[7]

    automaton = HexagonalAutomaton(nb_rows=nb_rows, nb_cols=nb_cols, p_melt=p_melt, p_freeze=p_freeze)

    # Set middle cell as the only active one
    automaton.grid = np.zeros((nb_rows, nb_cols), dtype=np.uint8)
    automaton.grid[(nb_rows // 2, nb_cols // 2)] = 1

    single_run(automaton, nb_frames=19)

    #for i in range(1):
    #    single_run(automaton, nb_frames=10
    #              render_path=str(render_dir / f"run_{i}.png"))


main(50, 50)
