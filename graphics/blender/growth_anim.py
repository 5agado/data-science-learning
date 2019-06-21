# Blender import system clutter
import bpy
import bmesh
import numpy as np
from typing import List

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import importlib
import ds_utils.blender_utils
importlib.reload(ds_utils.blender_utils)
from ds_utils.blender_utils import create_object


def get_particles():
    ref_obj = bpy.context.scene.objects['ref_obj']

    # If not called, there are no particles
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # Equivalent??
    par_system = ref_obj.evaluated_get(depsgraph).particle_systems[0]
    par_system = depsgraph.objects.get(ref_obj.name, None).particle_systems[0]

    # Reset memory.
    par_system.seed += 1
    par_system.seed -= 1

    particles = par_system.particles

    return particles


def shape_key_anim(objs_verts: List):
    obj = create_object(objs_verts[-1], edges=[], faces=[], obj_name="frame_{}".format(len(objs_verts)))
    objs_verts.pop()
    sk_basis = obj.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    obj.data.shape_keys.use_relative = False

    count = 1
    while True:
        if not objs_verts:
            break
        points = objs_verts.pop()

        # Create new shape-key block
        block = obj.shape_key_add(name=str(count), from_mix=False)  # returns a key_blocks member
        block.interpolation = 'KEY_LINEAR'
        block.value = 0

        # Update vertices position
        for (vert, co) in zip(block.data, points):
            vert.co = co

        for vert in block.data[len(points):]:
            vert.co = points[-1 ]

        # Keyframe evaluation time
        #bpy.context.object.active_shape_key_index = count+1
        obj.data.shape_keys.eval_time = count * 10
        obj.data.shape_keys.keyframe_insert(data_path='eval_time', frame=count*10)
        count += 1


def anim_objs(objs_verts: List):
    # Convert points to objects
    objs = [create_object(verts, edges=[], faces=[], obj_name="obj{}".format(i), collection='keyed_objs')
            for i, verts in enumerate(objs_verts)]

    # Base particle settings to use for the animation
    base_particle_settings = bpy.data.particles['base_particle_settings']

    # Reference object which holds the keyed particle system
    ref_obj = bpy.context.scene.objects['ref_obj']
    keyed_particle_system = ref_obj.particle_systems[0]

    for obj in objs:
        # create a copy of the base particle system
        obj.modifiers.new(obj.name, type='PARTICLE_SYSTEM')
        obj.particle_systems[0].settings = base_particle_settings

        # keyed the object into our reference
        bpy.ops.particle.new_target({'particle_system': keyed_particle_system})
        target = keyed_particle_system.active_particle_target
        target.object = obj


def anim_particles(objs_verts, nb_frames: int):
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nb_frames

    particles = get_particles()

    par_len = particles.data.settings.count

    par_birth = np.array([0.0] * par_len, dtype='float')
    particles.foreach_get('birth_time', par_birth)
    par_birth = np.ceil(par_birth)

    run_type = 0
    if run_type:
        for frame in range(0, nb_frames):
            bpy.context.scene.frame_set(frame)

            alive_p = [p for p in particles if p.alive_state == 'ALIVE']
            for p in alive_p:
                rand_obj = np.random.randint(frame+1)
                rand_loc = objs_verts[rand_obj][np.random.randint(len(objs_verts[rand_obj]))]
                p.location = rand_loc

            not_alive_p = [p for p in particles if p.alive_state == 'UNBORN']
            for p in not_alive_p:
                rand_loc = objs_verts[frame][np.random.randint(len(objs_verts[frame]))]
                p.location = rand_loc
    else:
        for frame in range(0, nb_frames):
            print("Updating growth anim for frame {}".format(frame))
            bpy.context.scene.frame_set(frame)

            target_obj = objs_verts[frame]
            par_index = np.where(par_birth == frame+1)
            target_obj_verts_rand_idxs = np.random.randint(0, len(target_obj), len(par_index[0]))
            for i, p_idx in enumerate(par_index[0]):
                particles[p_idx].location = target_obj[target_obj_verts_rand_idxs[i]]



    bpy.context.scene.frame_current = 0


def main():
    objs_verts = [[(1, 1, i), (1, -1, i), (-1, -1, i)] for i in range(5)]

    anim_particles(objs_verts, 5)

#test()
#main()