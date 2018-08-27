import bpy
import numpy as np
import pandas as pd


class Cube:
    def __init__(self, radius: float, location: tuple):
        loc = location

        verts = [
            (loc[0]+radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]+radius, loc[2]+radius),
            (loc[0]+radius, loc[1]-radius, loc[2]+radius),
            (loc[0]-radius, loc[1]-radius, loc[2]+radius),
            (loc[0]-radius, loc[1]+radius, loc[2]+radius),
        ]

        faces = [
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (4, 0, 3, 7)
        ]

        mesh_data = bpy.data.meshes.new("cube_mesh_data")
        mesh_data.from_pydata(verts, [], faces)
        mesh_data.update()

        obj = bpy.data.objects.new("Cube", mesh_data)

        scene = bpy.context.scene
        scene.objects.link(obj)


class Pyramid:
    def __init__(self, radius: float, location: tuple):
        loc = location

        verts = [
            (loc[0]+radius, loc[1]+radius, loc[2]-radius),
            (loc[0]+radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]-radius, loc[2]-radius),
            (loc[0]-radius, loc[1]+radius, loc[2]-radius),
            (loc[0], loc[1], loc[2]+radius),
        ]

        faces = [
            (0, 1, 2, 3),
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
        ]

        mesh_data = bpy.data.meshes.new("piramid_mesh_data")
        mesh_data.from_pydata(verts, [], faces)
        mesh_data.update()

        obj = bpy.data.objects.new("Piramid", mesh_data)

        scene = bpy.context.scene
        scene.objects.link(obj)


def cube_rec_shrink_step(cube: dict, depth: int = 0, max_depth=1):
    if depth >= max_depth:
        Cube(radius=cube['radius'], location=cube['location'])
        return
    else:
        radius = cube['radius']
        loc = cube['location']
        for i, x in enumerate(np.linspace(loc[0] - radius/2, loc[0] + radius/2, 3)):
            for j, y in enumerate(np.linspace(loc[1] - radius/2, loc[1] + radius/2, 3)):
                for k, z in enumerate(np.linspace(loc[2] - radius/2, loc[2] + radius/2, 3)):
                    if i == j == 1 or j == k == 1 or k == i == 1:
                        continue
                    else:
                        new_cube = {
                            'radius': radius/4,
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
        new_loc_top = (loc[0], loc[1], loc[2]+radius)
        new_loc_1 = (loc[0]+radius/2, loc[1]+radius/2, loc[2])
        new_loc_2 = (loc[0]-radius/2, loc[1]+radius/2, loc[2])
        new_loc_3 = (loc[0]+radius/2, loc[1]-radius/2, loc[2])
        new_loc_4 = (loc[0]-radius/2, loc[1]-radius/2, loc[2])
        new_locs = [new_loc_top, new_loc_1, new_loc_2, new_loc_3, new_loc_4]
        for new_loc in new_locs:
            new_pyramid = {
                'radius': radius/2,
                'location': new_loc
            }
            pyramid_rec_shrink_step(new_pyramid, depth + 1, max_depth)


def sierpinski_triangle(max_depth: int):
    object = {
        'radius': 10,
        'location': (0, 0, 0)
    }

    pyramid_rec_shrink_step(object, max_depth=max_depth)
    #cube_rec_shrink_step(object, max_depth=max_depth)

sierpinski_triangle(0)