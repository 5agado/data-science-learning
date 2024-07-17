"""
Setup to run synthetic-data-generation in Blender.
Requires to be run in a properly configured blend file
"""

import bpy  # Blender specific import
import numpy as np
from pathlib import Path
from datetime import datetime


def randomize_rgb(original_rgb, rand_magnitude=0.1):
    """ Add some noise to RGB and clip it back to [0,1] range """
    rand_rgb = original_rgb + np.random.uniform(low=-rand_magnitude, high=rand_magnitude, size=3)
    rand_rgb = rand_rgb.clip(0., 1.)
    return rand_rgb


def get_complementary_rgb(target_rgb: np.array):
    """ Get the complementary RGB """
    return 1 - target_rgb


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    rgb = [int(h[i:i + 2], 16)/255 for i in (0, 2, 4)]
    return np.array(rgb)


class RandColorGenerator:
    def __init__(self, colors_dist_threshold):
        self.colors_dist_threshold = colors_dist_threshold

    def get_random_colors(self, nb_colors):
        if nb_colors > 2:
            raise NotImplementedError('Method currently supports only single or pair colors')

        # generate first rand color
        colors = [np.random.uniform(low=0., high=1., size=3)]

        # if required, generate second color
        if nb_colors == 2:
            tries = 0
            dist = -1
            while dist < self.colors_dist_threshold:
                other_color = randomize_rgb(get_complementary_rgb(colors[0]))

                dist = np.linalg.norm(colors[0] - other_color)
                tries += 1
                if tries > 50:
                    raise Exception('Tried >50 times to generate second color. Exiting.')
            colors.append(other_color)
        return colors


def render_still(filepath: str, file_format='PNG'):
    """ Render a still frame for the current Blender scene """
    bpy.context.scene.render.filepath = filepath
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.ops.render.render(write_still=True)


def randomize_camera_loc(camera, xloc_range=None, yloc_range=None, zloc_range=None):
    """ Randomize camera location for each axis based on the provided ranges """
    new_loc = camera.location
    if xloc_range is not None:
        new_loc[0] = np.random.uniform(low=xloc_range[0], high=xloc_range[1])
    if yloc_range is not None:
        new_loc[1] = np.random.uniform(low=yloc_range[0], high=yloc_range[1])
    if zloc_range is not None:
        new_loc[2] = np.random.uniform(low=zloc_range[0], high=zloc_range[1])
    camera.location = new_loc


def randomize_world(color_value_range, bg_strength_range):
    """Randomize World setting, meaning background and ambient light"""
    # random world color value
    rand_value = np.random.uniform(low=color_value_range[0], high=color_value_range[1])
    bpy.data.worlds["World"].node_tree.nodes["RGB"].outputs[0].default_value = (rand_value, rand_value, rand_value, 1)

    # random world background strength
    rand_bg_strength = np.random.uniform(low=bg_strength_range[0], high=bg_strength_range[1])
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = rand_bg_strength


### Material Randomization

def randomize_colors(material, color_nodes_names):
    """ Randomize RGB nodes for the given Blender material """
    # apply colors in target material
    # naive way to generate multiple colors (as there is no guarantee of enough difference)
    for i, color_nodes_name in enumerate(color_nodes_names):
        rand_rgb = np.random.uniform(low=0., high=1., size=3)
        rand_rgb = np.append(rand_rgb, 1.)
        material.node_tree.nodes[color_nodes_name].outputs[0].default_value = tuple(rand_rgb)


def randomize_plain_pattern():
    material = bpy.data.materials["plain"]
    randomize_colors(material, ["RGB"])


def randomize_striped_pattern():
    material = bpy.data.materials["striped"]
    wave_texture = material.node_tree.nodes["Wave Texture"]
    wave_texture.inputs[1].default_value = np.random.uniform(low=2, high=12)  # rand scale
    wave_texture.bands_direction = np.random.choice(['Y', 'X'])  # rand bands direction

    randomize_colors(material, ["RGB", "RGB.001"])


def randomize_dotted_pattern(color_generator: RandColorGenerator):
    material = bpy.data.materials["dotted"]
    material.node_tree.nodes["Mapping"].inputs[2].default_value[2] = np.random.choice([0., 0.785398])  # rand rot of 45

    voronoi_texture = material.node_tree.nodes["Voronoi Texture"]
    voronoi_texture.inputs[2].default_value = np.random.uniform(low=6, high=40)  # rand scale
    voronoi_texture.inputs[5].default_value = np.random.uniform(low=0., high=0.2)  # rand uniformity

    randomize_colors(material, ["RGB", "RGB.001"])


def randomize_floral_pattern(texture_img_paths):
    material = bpy.data.materials["floral"]

    # load random texture image
    image = bpy.data.images.load(filepath=str(np.random.choice(texture_img_paths)))
    material.node_tree.nodes["Image Texture"].image = image

    # rand texture scale
    rand_scale = np.random.uniform(low=1.5, high=8)
    material.node_tree.nodes["Mapping"].inputs[3].default_value[0] = rand_scale
    material.node_tree.nodes["Mapping"].inputs[3].default_value[1] = rand_scale
    material.node_tree.nodes["Mapping"].inputs[3].default_value[2] = rand_scale

    randomize_colors(material, ["RGB"])


def get_images_paths(dirpath: Path):
    all_images_paths = []
    for img_type in ['jpg', 'png']:
        all_images_paths.extend(list(dirpath.glob(f'*.{img_type}')))
    return all_images_paths


def main(nb_runs: int, textures_path, output_path):
    """
    Run synthetic-data-generation process.
    For each target material and object randomize and render the scene for the given number of runs.
    """

    # create colors-generator to be used for each of the patterns
    rand_color_gen = RandColorGenerator(colors_dist_threshold=0.1)

    # each material to process needs a mapping between the existing Blender material name
    # and the corresponding randomization method
    target_materials = {
        'plain': lambda: randomize_plain_pattern(rand_color_gen),
        'striped': lambda: randomize_striped_pattern(rand_color_gen),
        'dotted': lambda: randomize_dotted_pattern(rand_color_gen),
        'floral': lambda: randomize_floral_pattern(get_images_paths(textures_path / 'floral'), rand_color_gen),
    }

    # target camera to use
    camera = bpy.data.objects['Camera']

    # run for each material
    for material_name, material_rand_fun in target_materials.items():
        print(f'Running material {material_name}')
        mat_out_path = output_path / material_name
        mat_out_path.mkdir(exist_ok=False, parents=True)

        # run for each object in the target collection
        for target_object in bpy.data.collections['target']:
            obj_name = target_object.name
            print(f'Running object {obj_name}')
            target_object.hide_render = False

            # apply the current material to the target object first material slot
            target_object.data.materials[0] = bpy.data.materials.get(material_name)

            # randomize before rendering
            for run in range(nb_runs):
                randomize_camera_loc(camera, xloc_range=[-3, 3], zloc_range=[1, 2])
                randomize_world(color_value_range=[0.7, 0.9], bg_strength_range=[0.2, 0.4])
                material_rand_fun()

                render_still(str(output_path / material_name / f'{obj_name}_run_{run:05}.png'))
            target_object.hide_render = True


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = Path.home() / f'sdg/pattern_{timestamp}'
    textures_path = Path.home() / f'sdg/textures'

    bpy.context.scene.eevee.taa_render_samples = 32
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = True

    main(nb_runs=2, textures_path=textures_path, output_path=out_path)
