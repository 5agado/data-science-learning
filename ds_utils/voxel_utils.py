import cv2
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

##################################
#    Frame edit functions
##################################

# b/w and resize
#lambda frame: cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255, frame_shape)