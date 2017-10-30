import numpy as np

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