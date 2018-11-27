import bpy
from bpy.app.handlers import persistent
import numpy as np
import pandas as pd

# Blender import system clutter
import sys
from pathlib import Path

SRC_PATH = Path.home() / "Documents/python_workspace/data-science-learning/graphics/heartbeat"
UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import heartbeat_utils
import importlib
importlib.reload(heartbeat_utils)


def duplicate_object(target, scene, material=None):
    new_obj = target.copy()
    new_obj.data = target.data.copy()
    new_obj.animation_data_clear()

    if material:
        new_obj.active_material = material.copy()

    scene.objects.link(new_obj)
    return new_obj


def create_grid(rows, cols):
    obj_grid = []
    scene = bpy.context.scene
    distance_factor = 4
    z = 1
    for i in range(rows):
        row = []
        for j in range(cols):
            cur_location = (i*distance_factor, j*distance_factor, z)
            cube = bpy.data.objects['base_cube']
            light = bpy.data.objects['internal_cube_light']
            cube_copy = duplicate_object(cube, scene)
            light_copy = duplicate_object(light, scene, material=light.active_material)
            cube_copy.location = cur_location
            light_copy.location = cur_location
            emission = light_copy.active_material.node_tree.nodes.get('Emission').inputs[1]
            row.append({'cube': cube_copy, 'light': light_copy, 'emission': emission})
        obj_grid.append(row)
    return obj_grid


def update_time_text(scene, test_data, time_text):
    time = test_data['index'].loc[scene.frame_current][10:16]
    time_text.data.body = time


def min_max_norm(vals: pd.Series, min_val=0., max_val=1.):
    return min_val + (((vals - vals.min()) * (max_val - min_val)) / (vals.max() - vals.min()))


def main():
    DAYS = 7
    WEEKS = 5
    NB_VALUES = 60*24  # minutes per day
    NUM_FRAMES = NB_VALUES

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES

    # Get HB data
    test_data = heartbeat_utils.load_test_data(str(Path.home() / "test_hb.csv"))

    # Create grid and base text
    obj_grid = create_grid(WEEKS, DAYS)

    # Setup text
    date_text = bpy.context.scene.objects['text_date']
    date_text.data.body = test_data['index'].loc[len(test_data)//2][:7]
    time_text = bpy.context.scene.objects['text_time']

    @persistent
    def update_time_handler(scene):
        update_time_text(scene, test_data, time_text)
    bpy.app.handlers.frame_change_pre.append(update_time_handler)

    # Normalize HB data
    min_val = 0.
    max_val = 2.
    test_data['value'] = min_max_norm(test_data['value'], min_val=min_val, max_val=max_val)

    # Animate
    for t in range(NB_VALUES):
        if t % 100 == 0:
            print("Updating frame {}".format(t))
        bpy.context.scene.frame_set(t)
        for k in range(WEEKS):
            for j in range(DAYS):
                cur_emission = obj_grid[k][j]['emission']
                cur_emission.default_value = test_data.loc[t+((DAYS*k+j)*NB_VALUES)].value
                cur_emission.keyframe_insert("default_value", frame=t)
    bpy.context.scene.frame_set(0)


main()
