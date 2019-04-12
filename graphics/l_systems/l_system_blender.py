# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "graphics/l_systems"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import LSystem
import importlib
importlib.reload(LSystem)
from LSystem import *

import bpy

from ds_utils.blender_utils import init_grease_pencil, draw_line


def lsystem_grid(layer_name='GP_layer'):
    print("###################################")

    max_depth = 4
    bpy.context.scene.frame_end = 100

    SPACING_FACTOR = 20
    NB_ROWS = 10
    NB_COLS = 8

    gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name=layer_name)
    gp_frame = gp_layer.frames.new(0)

    azimuth_adds = np.linspace(5, 70, NB_ROWS)
    inclination_adds = np.linspace(-20, 20, NB_COLS)

    for row in range(NB_ROWS):
        for col in range(NB_COLS):
            base_lsystem = LSystem(
                                axiom=['X'],
                                constants=['-', '+', '[', ']'],
                                rules={
                                    'X': 'F+[[X]-X]-F[-FX]+X',
                                    'F': 'FF',
                                },
                                render_config={
                                    'azimuth_add': azimuth_adds[row],
                                    'inclination_add': inclination_adds[col],
                                }
                            )

            print("###################")
            print(f"row {row}, col {col}")
            print(f"azimuth_add {base_lsystem.azimuth_add}, inclination_add {base_lsystem.inclination_add}")
            #print("n={}, alpha={}".format(max_depth, system.render_config['azimuth_add']))
            #print("{}".format(system.axiom))
            #print("{}".format(system.rules))
            for i in range(max_depth):
                if i%1 == 0:
                   print(f"Depth {i}")
                base_lsystem.rewrite()
            render(base_lsystem.current_state.copy(),
                   lambda x, y, context: draw(x, y, gp_layer, context),
                   azimuth_add=base_lsystem.azimuth_add, inclination_add=base_lsystem.inclination_add,
                   inclination=3.7,
                   pos=(row * SPACING_FACTOR, col * SPACING_FACTOR, 0),
                   line_length=.5)


def lsystem_params_anim(layer_name='GP_layer'):
    print("###################################")

    max_depth = 4
    nb_frames = 100
    bpy.context.scene.frame_end = nb_frames

    gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name=layer_name)

    azimuth_adds = np.linspace(-40, 40, nb_frames)
    inclination_adds = np.linspace(-20, 20, nb_frames)

    for frame in range(nb_frames):
        base_systems = [fractal_plant, fractal_plant_b, fractal_plant_c, fractal_plant_e, stochastic_plant]
        base_system = base_systems[np.random.randint(0, len(base_systems))]

        system = LSystem(
                            axiom=base_system.axiom,
                            constants=base_system.constants,
                            rules=base_system.rules,
                            rules_probs=base_system.rules_probs,
                            render_config={
                                'azimuth_add': azimuth_adds[np.random.randint(len(azimuth_adds))], #20.,
                                'inclination_add': inclination_adds[np.random.randint(len(inclination_adds))], #0.,
                            }
                        )

        print("###################")
        print(f"azimuth_add {system.azimuth_add}, inclination_add {system.inclination_add}")

        for i in range(max_depth):
            if i%1 == 0:
               print(f"Depth {i}")
            system.rewrite()
        gp_frame = gp_layer.frames.new(frame)
        render(system.current_state.copy(),
               lambda x, y, context: draw(x, y, gp_layer, context),
               azimuth_add=system.azimuth_add, inclination_add=system.inclination_add,
               inclination=3.7,
               pos=(0, 0, 0),
               line_length=1.)


def animate_lsystem(system: LSystem, max_depth: int, pos=(0, 0, 0), layer_name='GP_layer'):
    gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name=layer_name)
    print("###################")
    print(f"{layer_name}")
    print("n={}, alpha={}".format(max_depth, system.render_config['azimuth_add']))
    print(system.axiom)
    print(system.rules)
    for i in range(max_depth):
        if i%1 == 0:
           print(f"Depth {i}")
        gp_frame = gp_layer.frames.new(i)
        system.rewrite()  # notice we skip rendering axioms state
        render(system.current_state.copy(),
               lambda x, y, context: draw(x, y, gp_layer, context),
               azimuth_add=system.azimuth_add, inclination_add=system.inclination_add,
               inclination=4.,
               pos=(0, 0, 0),
               line_length=1.)


def draw(start: tuple, end: tuple, gp_layer, context):
    gp_frame = gp_layer.frames[-1]
    draw_line(gp_frame, start, end)


#lsystem_grid()

lsystem_params_anim()

#animate_lsystem(koch_curve, 4, layer_name='koch_curve')
#animate_lsystem(dragon_curve, 11, layer_name='dragon_curve')
#animate_lsystem(fractal_plant, 5, layer_name='fractal_plant')
#animate_lsystem(stochastic_plant, 5, layer_name='stochastic_plant')
#animate_lsystem(sierpinski_triangle, 7, layer_name='sierpinski_triangle')
#animate_lsystem(sierpinski_arrowhead_curve, 7, layer_name='sierpinski_arrowhead_curve')

