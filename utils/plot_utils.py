import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

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