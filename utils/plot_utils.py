import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def plot_sample_imgs(get_imgs_fun, img_shape, plot_side=5, savepath=None):
    """
    Generate visual samples and plot on a grid
    :param get_imgs_fun: function that given a int return a corresponding number of generated samples
    :param img_shape: shape of image to plot
    :param plot_side: samples per row (and column). Generated plot_side x plot_side samples.
    :param savepath: if given, save plot to such filepath, otherwise show plot
    """
    f, axarr = plt.subplots(plot_side, plot_side)
    samples = get_imgs_fun(plot_side*plot_side)
    for row in range(plot_side):
        for col in range(plot_side):
            axarr[row, col].imshow(samples[plot_side*row+col].reshape(img_shape))
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

#%matplotlib notebook
def animated_plot():
    fig, ax = sns.plt.subplots(dpi=100, figsize=(5, 4))
    sns.regplot(x, y, fit_reg=False)
    init_intercept, init_slope = res[0]
    line, = plt.plot([0, 1.0], [init_intercept, 1.0 * init_slope + init_intercept], 'k-')
    epoch_text = sns.plt.text(0, 0, "Epoch 0")
    sns.plt.show()

    def animate(i):
        current_intercept, current_slope = res[i]
        line.set_ydata([current_intercept, 1.0 * current_slope + current_intercept])
        epoch_text.set_text("Epoch {}, cost {:.3f}".format(i, history[i][0]))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(res)), interval=10)
    return ani


# Rotate tick labels
#ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=25)

# Save plot
#sns_plot.savefig("output.png")
#fig = swarm_plot.get_figure()
#fig.savefig(...)