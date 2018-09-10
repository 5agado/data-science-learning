import bpy
import sys
from mathutils import Vector
import numpy as np
from abc import ABC, ABCMeta, abstractmethod

# Scene parameters
SCALE_FACTOR = 0.05  # factor used to scale an object when close enough to target
ROTATION_VEC = (3.0, 0., 0.)  # rotation vector applied to an object when close enough to target
ORIGINAL_ROTATION_VEC = (0., 0., 0.)  # original rotation vector for an object
DIST_LIM = 6.5  # distance threshold for which an object is influenced by a target
TARGET_SPEED = 1.5  # speed at which the target is moving in the scene


# TODO
# Still some confusing naming around different "object" types.
# One cleanup option would be to embed everything in this class.
# The major thing to take into consideration is the difference between an actual 3D blender object
# and a basic data structure with object info. For example with the current duplication method a Blender
# object shares the same mesh data with all its duplicates.
class Object(ABC):
    sierpinski_scale = None  # scale to use during a recursive "sierpinski" step

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

    @staticmethod
    def scale_objects(object: dict, grid_val, scale_factor=SCALE_FACTOR):
        obj = object['object']
        origin_scale = object['orgn_scale']
        # grid value 1, object should end up with original size
        if grid_val:
            if obj.scale != origin_scale:
                obj.scale = origin_scale
        # grid value 0, object should end up scaled
        else:
            scaled_val = origin_scale * scale_factor
            if obj.scale != scaled_val:
                obj.scale = scaled_val
        # keyframe change
        obj.keyframe_insert("scale")

    @staticmethod
    def rotate_objects(object: dict, grid_val, rotation_vec=ROTATION_VEC, original_rot_vec=ORIGINAL_ROTATION_VEC):
        obj = object['object']
        # grid value 1, object should end up with original size
        if grid_val:
            if obj.rotation_euler != original_rot_vec:
                obj.rotation_euler = original_rot_vec
        # grid value 0, object should end up scaled
        else:
            if obj.rotation_euler != rotation_vec:
                obj.rotation_euler = rotation_vec
        # keyframe change
        obj.keyframe_insert("rotation_euler")

    @classmethod
    def obj_replication(cls, obj: dict, max_depth: int):
        """Entry point to manage Replication-Shrink for a target object"""
        object = cls(radius=obj['radius'], location=obj['location'])
        obj['object'] = object.obj

        sub_objs = [obj]
        for i in range(max_depth):
            new_sub_objs = []
            for sub_obj in sub_objs:
                new_sub_objs.extend(cls.replicate_shrink_step(sub_obj, i + 1))
                # delete original
                objs = bpy.data.objects
                objs.remove(sub_obj['object'], True)

            sub_objs = new_sub_objs
            # Scale mesh data (all copies should follow)
            for v in sub_objs[0]['object'].data.vertices:
                v.co *= cls.sierpinski_scale

        # Just at this point link object to scene
        for sub_obj in sub_objs:
            scene = bpy.context.scene
            scene.objects.link(sub_obj['object'])

        return sub_objs

    @classmethod
    @abstractmethod
    def replicate_shrink_step(cls, obj: dict, max_depth: int):
        """Replicates (mesh copy) the given object using "sierpinski" logic
            all the resulting sub-objects are then returned"""
        pass


class Cube(Object):
    sierpinski_scale = 1/3

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

    @classmethod
    def replicate_shrink_step(cls, cube: dict, max_depth: int):
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
                        # obj scaling (different from mesh one)
                        # keeps original dimensions, so need to keep track of depth
                        # cube_copy.scale = Vector((1 / 3**depth, 1 / 3**depth, 1 / 3**depth))
                        cube_copy.location = (x, y, z)
                        new_cube = {
                            'radius': radius * cls.sierpinski_scale,
                            'location': (x, y, z),
                            'object': cube_copy,
                            'orgn_scale': cube_copy.scale.copy()
                        }
                        sub_cubes.append(new_cube)
        return sub_cubes


class Pyramid(Object):
    sierpinski_scale = 1 / 2

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

        self.sierpinski_scale = 1 / 2

        self._init_mesh()
        self._init_obj()

    @classmethod
    def replicate_shrink_step(cls, pyramid: dict, depth: int):
        radius = pyramid['radius']
        loc = pyramid['location']
        pyramid_object = pyramid['object']
        sub_pyramids = []
        # amount of shifting for the center of new object
        center_shift = radius / 2
        # define the five locations for the five new sub-pyramids
        new_loc_top = (loc[0], loc[1], loc[2] + radius)
        new_loc_1 = (loc[0] + center_shift, loc[1] + center_shift, loc[2])
        new_loc_2 = (loc[0] - center_shift, loc[1] + center_shift, loc[2])
        new_loc_3 = (loc[0] + center_shift, loc[1] - center_shift, loc[2])
        new_loc_4 = (loc[0] - center_shift, loc[1] - center_shift, loc[2])
        new_locs = [new_loc_top, new_loc_1, new_loc_2, new_loc_3, new_loc_4]
        for new_loc in new_locs:
            pyramid_copy = pyramid_object.copy()
            # obj scaling (different from mesh one)
            # keeps original dimensions, so need to keep track of depth
            # pyramid_copy.scale = Vector((1 / 2**depth, 1 / 2**depth, 1 / 2**depth))
            pyramid_copy.location = new_loc
            new_pyramid = {
                'radius': radius * cls.sierpinski_scale,
                'location': new_loc,
                'object': pyramid_copy,
                'orgn_scale': pyramid_copy.scale.copy()
            }
            sub_pyramids.append(new_pyramid)

        return sub_pyramids


def update_grid(objs, target):
    target_loc = target.location
    for obj in objs:
        dist = np.linalg.norm(np.array(target_loc) - np.array(obj['location']))
        if dist < DIST_LIM:
            Object.scale_objects(obj, 0)
        else:
            Object.scale_objects(obj, 1)


# test method to move a target object along an axis
# to trigger updates to the sierpinski sub-objects in the scene
def move_target(target):
    (x, y, z) = target.location
    target.location = (x + np.random.rand() - 0.5,
                       y + np.random.rand() - 0.5,
                       z - TARGET_SPEED + np.random.rand() - 0.5)
    target.keyframe_insert("location")


# handler called at every frame change
def frame_handler(scene, objs, target, num_frames_change):
    frame = scene.frame_current
    if (frame % num_frames_change) == 0:
        move_target(target)
        # update grid
        update_grid(objs, target)


def main(_):
    NUM_FRAMES_CHANGE = 5  # higher values enable a more fluid transformation of objects, as frames between
    # keyframings interpolate the object modification taking place.

    bpy.ops.mesh.primitive_ico_sphere_add(
                       subdivisions=4,
                       size=0.3,
                       location=(0, 0, 30))

    target = bpy.context.scene.objects.active
    target.keyframe_insert("location")

    obj = {
        'location': (0, 0, 0),
        'radius': 10,
    }

    objs = Cube.obj_replication(obj, max_depth=2)
    #objs = Pyramid.obj_replication(obj, max_depth=3)

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, objs, target, NUM_FRAMES_CHANGE))


if __name__ == "__main__":
    main(sys.argv[1:])