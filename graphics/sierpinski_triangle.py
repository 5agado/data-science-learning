import bpy
import sys
from mathutils import Vector
import numpy as np
from abc import ABCMeta, abstractmethod


class Object:
    def __init__(self, radius: float, location: tuple, name: str):
        """
        Base utility class for a 3D object
        :param radius:
        :param location:
        :param name:
        """
        self.loc = location
        self.radius = radius
        self.name = name
        self.verts = []
        self.edges = []
        self.faces = []
        self.mesh_data = None
        self.obj = None

    def _init_mesh(self):
        mesh_data = bpy.data.meshes.new("{}_mesh_data".format(self.name))
        mesh_data.from_pydata(self.verts, self.edges, self.faces)
        mesh_data.update()
        self.mesh_data = mesh_data

    def _init_obj(self, link=True):
        self.obj = bpy.data.objects.new(self.name, self.mesh_data)

        if link:
            scene = bpy.context.scene
            scene.objects.link(self.obj)


class Cube(Object):
    def __init__(self, radius: float, location: tuple):
        super().__init__(radius=radius, location=location, name='cube')

        loc = location
        self.verts = [
            (loc[0]+radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]+radius, loc[2]+radius),
            (loc[0]+radius, loc[1]-radius, loc[2]+radius),
            (loc[0]-radius, loc[1]-radius, loc[2]+radius),
            (loc[0]-radius, loc[1]+radius, loc[2]+radius),
        ]

        self.faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (4, 0, 3, 7)
        ]

        self._init_mesh()
        self._init_obj()


class Pyramid(Object):
    def __init__(self, radius: float, location: tuple):
        super().__init__(radius=radius, location=location, name='pyramid')

        loc = location
        self.verts = [
            (loc[0]+radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]+radius, loc[2]-radius),
            (loc[0], loc[1], loc[2]+radius),
        ]

        self.faces = [
            (0, 1, 2, 3),
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ]

        self._init_mesh()
        self._init_obj()


# Recursive Shrink Step
#
# This methods recursively apply a "sierpinski" step for the specific object
# and once reached the given max depth proceed to create an instance of the current object
# which includes info about location and size (radius)


def cube_rec_shrink_step(cube: dict, depth: int = 0, max_depth=1):
    if depth >= max_depth:
        Cube(radius=cube['radius'], location=cube['location'])
        return
    else:
        radius = cube['radius']
        loc = cube['location']
        # amount of shifting for the center of new object
        center_shift = radius*(2/3)
        # create three equally spaced sub-cubes for each dimension
        for i, x in enumerate(np.linspace(loc[0] - center_shift, loc[0] + center_shift, 3)):
            for j, y in enumerate(np.linspace(loc[1] - center_shift, loc[1] + center_shift, 3)):
                for k, z in enumerate(np.linspace(loc[2] - center_shift, loc[2] + center_shift, 3)):
                    if i == j == 1 or j == k == 1 or k == i == 1:
                        continue
                    else:
                        new_cube = {
                            'radius': radius/3,
                            'location': (x, y, z)
                        }
                        cube_rec_shrink_step(new_cube, depth + 1, max_depth)


def pyramid_rec_shrink_step(pyramid: dict, depth: int = 0, max_depth=1):
    if depth >= max_depth:
        Pyramid(radius=pyramid['radius'], location=pyramid['location'])
        return
    else:
        radius = pyramid['radius']
        loc = pyramid['location']
        # amount of shifting for the center of new object
        center_shift = radius/2
        # define the five locations for the five new sub-pyramids
        new_loc_top = (loc[0], loc[1], loc[2]+radius)
        new_loc_1 = (loc[0]+center_shift, loc[1]+center_shift, loc[2])
        new_loc_2 = (loc[0]-center_shift, loc[1]+center_shift, loc[2])
        new_loc_3 = (loc[0]+center_shift, loc[1]-center_shift, loc[2])
        new_loc_4 = (loc[0]-center_shift, loc[1]-center_shift, loc[2])
        new_locs = [new_loc_top, new_loc_1, new_loc_2, new_loc_3, new_loc_4]
        for new_loc in new_locs:
            new_pyramid = {
                'radius': radius/2,
                'location': new_loc
            }
            pyramid_rec_shrink_step(new_pyramid, depth + 1, max_depth)


# Replication Shrink Step
#
# This methods replicates (mesh copy) the given object using "sierpinski" logic
# all the resulting sub-objects are then returned


def cube_replicate_shrink_step(cube: dict, depth: int):
    radius = cube['radius']
    loc = cube['location']
    cube_obj = cube['object']
    sub_cubes = []
    # amount of shifting for the center of new object
    center_shift = radius * (2 / 3)
    for i, x in enumerate(np.linspace(loc[0] - center_shift, loc[0] + center_shift, 3)):
        for j, y in enumerate(np.linspace(loc[1] - center_shift, loc[1] + center_shift, 3)):
            for k, z in enumerate(np.linspace(loc[2] - center_shift, loc[2] + center_shift, 3)):
                if i == j == 1 or j == k == 1 or k == i == 1:
                    continue
                else:
                    cube_copy = cube_obj.copy()
                    # TODO fix this
                    # ideally: resize original (all should follow)
                    # right now: keep original dimensions, so need to keep track of depth
                    cube_copy.scale = Vector((1 / 3**depth, 1 / 3**depth, 1 / 3**depth))
                    cube_copy.location = (x, y, z)
                    new_cube = {
                        'radius': radius / 3,
                        'location': (x, y, z),
                        'object': cube_copy,
                        'orgn_scale': cube_copy.scale.copy()
                    }
                    sub_cubes.append(new_cube)
                    scene = bpy.context.scene
                    scene.objects.link(cube_copy)
    #scale = 1/3
    #for v in cube_obj.data.verts:
    #    v.co *= scale
    # delete original
    objs = bpy.data.objects
    objs.remove(cube_obj, True)
    return sub_cubes


def pyramid_replicate_shrink_step(pyramid: dict, depth: int):
    radius = pyramid['radius']
    loc = pyramid['location']
    pyramid_object = pyramid['object']
    sub_pyramids = []
    # amount of shifting for the center of new object
    center_shift = radius/2
    # define the five locations for the five new sub-pyramids
    new_loc_top = (loc[0], loc[1], loc[2] + radius)
    new_loc_1 = (loc[0] + center_shift, loc[1] + center_shift, loc[2])
    new_loc_2 = (loc[0] - center_shift, loc[1] + center_shift, loc[2])
    new_loc_3 = (loc[0] + center_shift, loc[1] - center_shift, loc[2])
    new_loc_4 = (loc[0] - center_shift, loc[1] - center_shift, loc[2])
    new_locs = [new_loc_top, new_loc_1, new_loc_2, new_loc_3, new_loc_4]
    for new_loc in new_locs:
        pyramid_copy = pyramid_object.copy()
        # TODO fix this
        # ideally: resize original (all should follow)
        # right now: keep original dimensions, so need to keep track of depth
        pyramid_copy.scale = Vector((1 / 2**depth, 1 / 2**depth, 1 / 2**depth))
        pyramid_copy.location = new_loc
        new_cube = {
            'radius': radius / 2,
            'location': new_loc,
            'object': pyramid_copy,
            'orgn_scale': pyramid_copy.scale.copy()
        }
        sub_pyramids.append(new_cube)
        scene = bpy.context.scene
        scene.objects.link(pyramid_copy)
    #scale = 1/3
    #for v in cube_obj.data.verts:
    #    v.co *= scale
    # delete original
    objs = bpy.data.objects
    objs.remove(pyramid_object, True)
    return sub_pyramids


# Entry point to call a Recursive-Shrink-Step for a target object
def rec_shrink(max_depth: int):
    obj = {
        'radius': 10,
        'location': (0, 0, 0)
    }

    #pyramid_rec_shrink_step(obj, max_depth=max_depth)
    cube_rec_shrink_step(obj, max_depth=max_depth)


# Entry point to manage Replication-Shrink for a target object
def obj_replication(max_depth: int, use_pyramid=False):
    obj = {
        'radius': 10,
        'location': (0, 0, 0),
        'orgn_scale': Vector((1.0, 1.0, 1.0))
    }

    if use_pyramid:
        # Pyramid case
        pyramid = Pyramid(radius=obj['radius'], location=obj['location'])
        obj['object'] = pyramid.obj
    else:
        # Cube case
        cube = Cube(radius=obj['radius'], location=obj['location'])
        obj['object'] = cube.obj

    sub_objs = [obj]
    for i in range(max_depth):
        new_sub_objs = []
        for sub_obj in sub_objs:
            if use_pyramid:
                new_sub_objs.extend(pyramid_replicate_shrink_step(sub_obj, i + 1))
            else:
                new_sub_objs.extend(cube_replicate_shrink_step(sub_obj, i + 1))
        sub_objs = new_sub_objs
    return sub_objs


def scale_objects(object, grid_val, scale_factor=0.2, keyframe=True):
    obj = object['object']
    origin_scale = object['orgn_scale']
    # grid value 1, object should end up with original size
    if grid_val:
        # skip all (keyframing too) if already ok, otherwise set original size
        # TODO this if else check cause some error where the
        # scaling get scrambled once animation is stopped
        #if obj.scale == origin_scale:
        #    return
        #else:
        obj.scale = origin_scale.copy()
    # grid value 0, object should end up scaled
    else:
        # skip all (keyframing too) if already ok, otherwise set scaled size
        #if obj.scale == origin_scale*scale_factor:
        #    return
        #else:
        obj.scale = origin_scale*scale_factor
    if keyframe:
        obj.keyframe_insert("scale")


def update_grid(objs, target):
    DIST_LIM = 6.5
    target_loc = target.location
    for obj in objs:
        dist = np.linalg.norm(np.array(target_loc) - np.array(obj['location']))
        if dist < DIST_LIM:
            scale_objects(obj, 0)
        else:
            scale_objects(obj, 1)


# handler called at every frame change
def frame_handler(scene, objs, target, num_frames_change):
    TARGET_SPEED = 0.5
    frame = scene.frame_current
    n = frame % num_frames_change
    if n == 0:
        # move target
        (x, y, z) = target.location
        #target.location = (x-TARGET_SPEED, y, z)
        target.location = (x, y, z - TARGET_SPEED)
        target.keyframe_insert("location")
        # update grid
        update_grid(objs, target)


def main(_):
    # CONSTANTS
    num_frames_change = 2

    #bpy.ops.mesh.primitive_ico_sphere_add(
    #                    subdivisions = 4,
    #                    size = 1,
    #                    location = (25, 0, 0))
    bpy.ops.mesh.primitive_ico_sphere_add(
                       subdivisions = 4,
                       size = 0.3,
                       location = (0, 0, 30))

    target = bpy.context.scene.objects.active
    target.keyframe_insert("location")

    objs = obj_replication(4, use_pyramid=True)
    # rec_shrink(1)
    update_grid(objs, target)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, objs, target, num_frames_change))


if __name__ == "__main__":
    main(sys.argv[1:])