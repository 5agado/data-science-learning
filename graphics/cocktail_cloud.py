# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import bpy
import bmesh
from math import cos, sin, pi
import numpy as np


def generate_lightning_effect(obj, container, nb_flashes: int, nb_frames: int,
                              max_energy: int, min_energy: int, start_frame=0):
    # bounding box
    x_radius = container.scale.x/2
    y_radius = container.scale.y/2
    z_radius = container.scale.z/2
    x_lim = (container.location.x - x_radius, container.location.x + x_radius)
    y_lim = (container.location.y - y_radius, container.location.y + y_radius)
    z_lim = (container.location.z - z_radius, container.location.z + z_radius)

    # init location and energy
    set_obj_flash(obj, loc=(0, 0, 0), energy=0., frame=0)

    frame = start_frame
    avg_flash_interval = nb_frames // (nb_flashes + 1)
    for i in range(nb_flashes):
        # location
        new_loc = (np.random.uniform(x_lim[0], x_lim[1]),
                   np.random.uniform(y_lim[0], y_lim[1]),
                   np.random.uniform(z_lim[0], z_lim[1]))

        # flash energy
        energy = np.random.uniform(min_energy, max_energy)

        # frame
        frame += avg_flash_interval

        set_obj_flash(obj, loc=new_loc, energy=0., frame=frame - np.random.randint(1, 3))   # set at new location
        set_obj_flash(obj, loc=new_loc, energy=energy, frame=frame)                         # flash
        #new_loc = np.array(new_loc) + np.random.rand(3)
        set_obj_flash(obj, loc=new_loc, energy=0., frame=frame + np.random.randint(3, 10))  # die off


def set_obj_flash(obj, loc, energy: float, frame: int):
    bpy.context.scene.frame_set(frame)

    obj.location = loc
    obj.keyframe_insert("location")
    obj.data.energy = energy
    obj.data.keyframe_insert("energy")


#for a in bpy.data.actions:
#    bpy.data.actions.remove(a)
obj = bpy.context.scene.objects['lightning_01']
obj2 = bpy.context.scene.objects['lightning_02']
container = bpy.context.scene.objects['light_container']
generate_lightning_effect(obj, container, nb_flashes=10, nb_frames=200,
                          max_energy=6000, min_energy=5500, start_frame=40)
generate_lightning_effect(obj2, container, nb_flashes=8, nb_frames=200,
                          max_energy=8500, min_energy=7500, start_frame=40)
bpy.context.scene.frame_set(0)