from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime

# basic helper class to sample random noise
class NoiseDistribution:
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self.dist_fun = np.random.normal

    def sample(self, shape):
        return self.dist_fun(size=shape)

# utility to set if net is trainable or not
# ??For Keras, need to recompile in order to actuate the changes?
def set_trainable(net, val, loss=None, optimizer=None):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val
    #net.compile(loss=loss, optimizer=optimizer)


###########################
# LATENT SPACE EXPLORATION
###########################

# Requires
# plt.rcParams['animation.ffmpeg_path'] = str(Path.home() / "anaconda3/envs/image-processing/bin/ffmpeg")

def animate_latent_transition(latent_vectors, gen_image_fun, gen_latent_fun,
                              img_size: tuple, nb_frames: int,
                              img_is_bw=False, render_dir: Path = None,
                              fps=30):
    # setup plot
    dpi = 100
    fig, ax = plt.subplots(dpi=dpi, figsize=(img_size[0] / dpi, img_size[1] / dpi))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Need first gen, otherwise if filled with zeros/ones can't see results (probably different dtype init)
    im = ax.imshow(gen_image_fun(latent_vectors), cmap='gray' if img_is_bw else 'jet')
    plt.axis('off')

    def animate(i, latent_vectors, gen_image_fun, gen_latent_fun):
        current_latents = gen_latent_fun(latent_vectors, i)
        im.set_data(gen_image_fun(current_latents))

    ani = animation.FuncAnimation(fig, animate, frames=nb_frames, interval=1,
                                  fargs=[latent_vectors, gen_image_fun, gen_latent_fun])

    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
        ani.save(str(render_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + '.mp4')),
                 animation.FFMpegFileWriter(fps=fps))


def gen_latent_linear(latent_vectors, frame_idx, nb_points):
    """
    Interpolate linearly between the given vectors, for nb_points values
    :param latent_vectors:
    :param frame_idx:
    :param nb_points:
    :return:
    """
    latent_start = latent_vectors[frame_idx // nb_points]
    latent_end = latent_vectors[frame_idx // nb_points + 1]
    latent_diff = latent_end - latent_start
    latent_vec = latent_start + (latent_diff / nb_points) * (frame_idx % nb_points)
    return np.array([latent_vec])


def gen_latent_idx(latent_vectors, frame_idx, vec_idx, vals):
    """
    Set vec_idx value of latent vector to frame_idx value of the given vals
    :param latent_vectors:
    :param frame_idx:
    :param vec_idx:
    :param vals:
    :return:
    """
    latent_vec = latent_vectors[0].copy()
    latent_vec[vec_idx] = vals[frame_idx]
    return np.array([latent_vec])
