import bpy
import bmesh
from mathutils import Vector
import numpy as np

# Blender import system clutter
import sys
from pathlib import Path

SRC_PATH = Path.home() / "Documents/python_workspace/data-science-learning/graphics/agents"
UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import Flock
import importlib
importlib.reload(Flock)
from Flock import Flock
from utils.blender_utils import delete_all

compute_animation = False
sphere_subdivisions = 3

Flock.NB_DIMENSIONS = 3

Flock.COHESION_FACTOR = 1 / 5
Flock.ALIGNMENT_FACTOR = 1 / 5
Flock.SEPARATION_FACTOR = 1 / 2
Flock.ATTRACTOR_FACTOR = 1 / 20
Flock.VELOCITY_FACTOR = 2

Flock.COHESION = True
Flock.ALIGNMENT = True
Flock.ATTRACTOR = True
Flock.SEPARATION = True

DRAW_VISIBILITY = False
DRAW_VELOCITY = False
TARGET_SPEED = 1.

bl_info = {
    "name": "Flock Animation",
    "author": "Alex Martinelli",
    "location": "View3D > Tools > Flock",
    "version": (1, 0, 0),
    "blender": (2, 7, 8),
    "description": "Flock Animation",
    "category": "Development"
}


# Panel takes care of the UI components
class SimplePanel(bpy.types.Panel):
    # Hierarchically define location of the add-on in the Blender UI
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_category = "Flock"
    bl_label = "Create Flock"

    # define panel UI components
    def draw(self, context):
        # sample button
        self.layout.operator("object.create_flock", text="Create Flock")

        self.layout.prop(context.scene, 'FLOCK_SIZE')
        self.layout.prop(context.scene, 'NUM_FRAMES_CHANGE')
        self.layout.prop(context.scene, 'NUM_FRAMES')
        self.layout.prop(context.scene, 'SEED')
        self.layout.prop(context.scene, 'SPHERE_SIZE')
        self.layout.prop(context.scene, 'VOLUME_SIZE')
        self.layout.prop(context.scene, 'ATTRACTOR_CONFINE')
        self.layout.prop(context.scene, 'VISIBILITY_RADIUS')
        self.layout.prop(context.scene, 'CLOSENESS')
        self.layout.prop(context.scene, 'ANIMATE')

    @classmethod
    def register(cls):
        print("Registered class: %s " % cls.bl_label)
        # Register properties related to the class here.


    @classmethod
    def unregister(cls):
        print("Unregistered class: %s " % cls.bl_label)
        # Delete parameters related to the class here


# Operator is the actual logic behind the add-of
class SimpleOperator(bpy.types.Operator):
    bl_idname = "object.create_flock"
    bl_label = "Create Flock"

    def execute(self, context):
        delete_all()

        sphere_size = context.scene.SPHERE_SIZE
        volume_size = context.scene.VOLUME_SIZE

        Flock.VISIBILITY_RADIUS = context.scene.VISIBILITY_RADIUS
        Flock.CLOSENESS = context.scene.CLOSENESS
        Flock.ATTRACTOR_CONFINE = context.scene.ATTRACTOR_CONFINE

        flock = init_flock(context.scene.FLOCK_SIZE,
                           sphere_size=sphere_size,
                           seed=context.scene.SEED,
                           volume_size=volume_size)

        bpy.context.scene.frame_end = context.scene.NUM_FRAMES

        # Animate directly by looping through frames
        if compute_animation:
            for i in range(context.scene.NUM_FRAMES):
                if i % 10 == 0:
                    print("Updating frame {}".format(i))
                bpy.context.scene.frame_set(i)
                frame_handler(bpy.context.scene, flock, context)

        else:
            # Animate by adding handler to frame change
            bpy.app.handlers.frame_change_pre.clear()
            bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, flock, context))

        # better to return this string when done with the execution work
        return {'FINISHED'}

    @classmethod
    def register(cls):
        print("Registered operator: %s " % cls.bl_label)
        # Register properties related to the class here
        bpy.types.Scene.FLOCK_SIZE = bpy.props.IntProperty(name="Flock Size",
                                                           description="Number of units in the flock",
                                                           default=15,
                                                           min=1,
                                                           max=500)
        bpy.types.Scene.NUM_FRAMES_CHANGE = bpy.props.IntProperty(name="Num Frames Change",
                                                                  description="Number of frames required to trigger a change",
                                                                  default=2,
                                                                  min=1,
                                                                  max=20)
        bpy.types.Scene.NUM_FRAMES = bpy.props.IntProperty(name="Num Frames",
                                                           description="Number of frames for the animation",
                                                           default=250,
                                                           min=10,
                                                           max=500)
        bpy.types.Scene.SEED = bpy.props.IntProperty(name="Seed",
                                                     description="Random seed",
                                                     default=42,
                                                     min=0,
                                                     max=100)
        bpy.types.Scene.VISIBILITY_RADIUS = bpy.props.IntProperty(name="Visibility Radius",
                                                                  description="Visibility Radius",
                                                                  default=7,
                                                                  min=1,
                                                                  max=100)
        bpy.types.Scene.CLOSENESS = bpy.props.IntProperty(name="Closeness",
                                                          description="Closeness",
                                                          default=1,
                                                          min=0,
                                                          max=100)
        bpy.types.Scene.ATTRACTOR_CONFINE = bpy.props.IntProperty(name="Attractor Confine",
                                                                  description="Attractor Confine",
                                                                  default=7,
                                                                  min=1,
                                                                  max=100)
        bpy.types.Scene.VOLUME_SIZE = bpy.props.IntProperty(name="Volume Size",
                                                                  description="Volume Size",
                                                                  default=10,
                                                                  min=1,
                                                                  max=100)
        bpy.types.Scene.SPHERE_SIZE = bpy.props.FloatProperty(name="Sphere Size",
                                                              description="Sphere Size",
                                                              default=0.2,
                                                              min=0.1,
                                                              max=10.)

        def animate_bool(_1, _2):
            global compute_animation
            compute_animation = not compute_animation
        bpy.types.Scene.ANIMATE = bpy.props.BoolProperty(update=animate_bool)


    @classmethod
    def unregister(cls):
        print("Unregistered operator: %s " % cls.bl_label)
        # Delete parameters related to the class here


def register():

    # Implicitly register objects inheriting bpy.types in current file and scope
    #bpy.utils.register_module(__name__)

    # Or explicitly register objects
    bpy.utils.register_class(SimpleOperator)
    bpy.utils.register_class(SimplePanel)

    print("%s registration complete\n" % bl_info.get('name'))


def unregister():
    # Always unregister in reverse order to prevent error due to
    # interdependencies

    # Explicitly unregister objects
    bpy.utils.unregister_class(SimpleOperator)
    bpy.utils.unregister_class(SimplePanel)

    # Or unregister objects inheriting bpy.types in current file and scope
    #bpy.utils.unregister_module(__name__)

    print("%s unregister complete\n" % bl_info.get('name'))


# Called only when running the script from Blender
# when distributed as plugin register() and unregister() are used
if __name__ == "__main__":

    try:
        unregister()
    except Exception as e:
        print(e)
        pass

    register()


def create_line():
    me = bpy.data.meshes.new('line')
    verts = [(1, 1, 1), (2, 2, -1)]
    edges = [(0, 1)]

    me.from_pydata(verts, edges, [])

    # Add the mesh to the scene
    scene = bpy.context.scene
    obj = bpy.data.objects.new("Line", me)
    scene.objects.link(obj)
    return obj


def create_circle():
    # Make a new BMesh
    bm = bmesh.new()

    # Add a circle XXX, should return all geometry created, not just verts.
    bmesh.ops.create_circle(
        bm,
        cap_ends=False,
        diameter=Flock.VISIBILITY_RADIUS,
        segments=8)

    me = bpy.data.meshes.new("Mesh")
    bm.to_mesh(me)
    bm.free()

    # Add the mesh to the scene
    scene = bpy.context.scene
    obj = bpy.data.objects.new("Visibility", me)
    scene.objects.link(obj)
    return obj


def init_flock(size: int, sphere_size: float, seed: int = None, volume_size: int = 1):
    ATTRACTOR_POS = (0, 0, 30)

    flock = Flock(size, canvas_size=volume_size, canvas_shift=(-volume_size//2, -volume_size//2,
                                                               ATTRACTOR_POS[2]-volume_size//2),
                  seed=seed)

    # Add object for attractor
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=ATTRACTOR_POS)
    flock.attractor_obj = bpy.context.scene.objects.active
    flock.attractor_pos = np.array(flock.attractor_obj.location)

    # Create basic sphere from which we will copy the mesh
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=sphere_subdivisions,
        size=sphere_size,
        location=(0, 0, 0))
    base_obj = bpy.context.scene.objects.active
    scene = bpy.context.scene

    for unit in flock.units:
        # add blender object to unit attributes
        unit.obj = base_obj.copy()
        unit.obj.location = unit.pos
        scene.objects.link(unit.obj)
        # add visibility border
        if DRAW_VISIBILITY:
            unit.visibility = create_circle()
            unit.visibility.location = unit.pos
        # add velocity line
        if DRAW_VELOCITY:
            unit.vel_obj = create_line()
            unit.vel_obj.data.vertices[0].co = unit.pos
            unit.vel_obj.data.vertices[1].co = unit.pos+unit.vel*4

    # delete original
    objs = bpy.data.objects
    objs.remove(base_obj, True)

    return flock


def move_target(target, frame: int):
    (x, y, z) = target.location
    if frame < 100:
        target.location = (x + np.random.rand() - 0.5,
                           y + np.random.rand() - 0.5,
                           z - TARGET_SPEED + np.random.rand() - 0.5)
    elif 100 <= frame < 150:
        target.location = (x + np.random.rand() - 0.5,
                           y + np.random.rand() - 0.5,
                           z + TARGET_SPEED + np.random.rand() - 0.5)
    elif frame >= 150:
        target.location = (x + np.random.rand() - 0.5,
                           y + TARGET_SPEED + np.random.rand() - 0.5,
                           z + np.random.rand() - 0.5)
    target.keyframe_insert("location")


# handler called at every frame change
def frame_handler(scene, flock: Flock, context):
    frame = scene.frame_current
    # When reaching final frame, clear handlers
    if frame >= context.scene.NUM_FRAMES:
        bpy.app.handlers.frame_change_pre.clear()
    elif (frame % context.scene.NUM_FRAMES_CHANGE) == 0:
        flock.update()
        move_target(flock.attractor_obj, frame)
        flock.attractor_pos = np.array(flock.attractor_obj.location)
        for unit in flock.units:
            # update and keyframe location
            unit.obj.location = unit.pos
            unit.obj.keyframe_insert("location")
            # update visibility position
            if DRAW_VISIBILITY:
                unit.visibility.location = unit.pos
                unit.visibility.keyframe_insert("location")
            # update velocity line
            if DRAW_VELOCITY:
                unit.vel_obj.data.vertices[0].co = unit.pos
                unit.vel_obj.data.vertices[1].co = unit.pos + unit.vel * 4