import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def create_voxel_data(video_path: str, frame_edit_fun=None, out_path=None, nb_repeat_frame=1):
    voxel_data = load_video_as_numpy(video_path=video_path, frame_edit_fun=frame_edit_fun,
                                     nb_repeat_frame=nb_repeat_frame)
    print("Shape ", str(voxel_data.shape))

    # save as _8bit.raw
    if out_path:
        with open(out_path, 'wb') as f:
            voxel_data = voxel_data.reshape(np.prod(voxel_data.shape), order='A')
            voxel_data.tofile(f)

    return voxel_data


from mpl_toolkits.mplot3d import Axes3D
def plot_voxel_data(data):
    # Example data generation
    #data_shape = (6, 6, 6)
    #rand_grid = np.random.rand(*data_shape)
    #voxel_grid = rand_grid > 0.5

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(data)

    plt.show()


def load_video_as_numpy(video_path: str, frame_edit_fun=None, nb_repeat_frame=1):
    input_video = cv2.VideoCapture(video_path)
    frames = []
    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret:
            if frame_edit_fun:
                frame = frame_edit_fun(frame)
            frames.extend([frame] * nb_repeat_frame)
        else:
            break

    return np.array(frames)


def get_sphere_mask(shape, radius: int):
    """
    Get spherical mask for the given volume (centered in the volume center and of the given radius).
    Points outside the sphere will be set to zero
    :param shape:
    :param radius:
    :return:
    """
    mask = np.zeros(shape)

    # volume center
    x0, y0, z0 = int(np.floor(shape[0] / 2)), int(np.floor(shape[1] / 2)), int(np.floor(shape[2] / 2))

    # check for all relevant point if inside or outside the sphere
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                # deb>=0: inside the sphere, deb<0: outside the sphere
                deb = radius - math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2 + (z0 - z) ** 2)
                if deb >= 0:
                    mask[x, y, z] = 1
    return mask


def get_volume_coordinates(size: int, scale=1., translation=0.):
    # get pixels coordinates in the range [-1, 1]
    # this would be equivalent to explicitly operating min-max normalization
    x_range = scale * np.linspace(-1., 1., size) + translation
    y_range = scale * np.linspace(-1., 1., size) + translation
    z_range = scale * np.linspace(-1., 1., size) + translation

    # repeat each range along the opposite axis
    g = np.meshgrid(x_range, y_range, z_range)

    # zip coordinates, shape=(size**3, 3)
    all_xyz = np.array(list(zip(*(v.flat for v in g))))

    return all_xyz


##################################
#    Frame edit functions
##################################

# b/w and resize
#lambda frame: cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255, frame_shape)

##################################
#    Marching Cubes
##################################

#import mcubes
#vertices, triangles = mcubes.marching_cubes(volume, threshold)
#mcubes.export_obj(vertices, triangles, out_path / 'test.obj')