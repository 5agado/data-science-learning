import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


def plot_sample_imgs(get_imgs_fun, img_shape, plot_side=5, savepath=None, cmap='gray'):
    """
    Generate visual samples and plot on a grid
    :param get_imgs_fun: function that given a int return a corresponding number of generated samples
    :param img_shape: shape of image to plot
    :param plot_side: samples per row (and column). Generated plot_side x plot_side samples
    :param savepath: if given, save plot to such filepath, otherwise show plot
    :param cmap: matplotlib specific cmap to use for the plot
    """
    f, axarr = plt.subplots(plot_side, plot_side)
    samples = get_imgs_fun(plot_side*plot_side)
    for row in range(plot_side):
        for col in range(plot_side):
            axarr[row, col].imshow(samples[plot_side*row+col].reshape(img_shape), cmap=cmap)
            axarr[row, col].set_title('')
            axarr[row, col].axis('off')
    if savepath:
        f.savefig(savepath)
        plt.close()
    else:
        plt.show()


def plot_correlation(df):
    # Correlation
    corr = df.corr()
    print(corr)
    # Plot masking the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask)
    sns.plt.show()


"""
# Rotate tick labels
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=25)

# Save plot
sns_plot.savefig("output.png")
fig = swarm_plot.get_figure()
fig.savefig(...)

# Matplotlib to Plotly
import plotly.tools as tls
plotly_fig = tls.mpl_to_plotly(mpl_fig)
py.iplot(plotly_fig
"""

##############################
#        Animation
##############################

#%matplotlib notebook  # rely on notebook mode as the inline doesn't seem to work in Jupyter
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '~/path/to/bin/ffmpeg'


def animated_plot(img_width: int, img_height: int, nb_frames: int, outpath: str = None):
    # Setup plot
    dpi = 100
    if outpath:
        fig, ax = plt.subplots(dpi=dpi, figsize=(img_width / dpi, img_height / dpi))
    else:
        fig, ax = plt.subplots(dpi=dpi, figsize=(5, 5))
    plt.axis('off')

    #line, = plt.plot([0, 1.0], [init_intercept, 1.0 * init_slope + init_intercept], 'k-')
    #epoch_text = plt.text(0, 0, "Epoch 0")
    #im = ax.imshow(np.zeros((28, 28)), cmap='gray')

    def animate(i, ):
        pass
        #current_intercept, current_slope = res[i]
        #line.set_ydata([current_intercept, 1.0 * current_slope + current_intercept])
        #epoch_text.set_text("Epoch {}, cost {:.3f}".format(i, history[i][0]))
        #return line,
        # one other option is to set the data like
        #im.set_data(np.zeros((28, 28))+1)
        #ax.imshow(system.B, cmap='gray')

    # Animate
    ani = animation.FuncAnimation(fig, animate, frames=nb_frames, interval=100,
                                  fargs=[])  # be sure to pass the additional args needed for the animation

    if outpath:
        ani.save(outpath, animation.FFMpegFileWriter(fps=30))
    else:
        return ani

"""
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30)
animation.writers.list()
"""

##############################
#        Drawing
##############################

def draw_template():
    from PIL import Image, ImageDraw
    img_size = 1000
    img = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 20, 180, 180), fill='blue', outline='blue')