# Heavy refactor of https://github.com/MNoichl/physarum

#import numpy as np
import cupy as np
from PIL import Image as IMG
from typing import List, Tuple
import math
import tqdm

from cupyx.scipy.ndimage.filters import gaussian_filter, uniform_filter, median_filter
#from scipy.ndimage.filters import gaussian_filter, uniform_filter, median_filter
import noise

class Physarum:
    """A class that contains the parameters that set up a physarum-population and keep track of its development."""

    def __init__(self,
                 shape: Tuple,
                 horizon_walk=10.0,
                 horizon_sense=40.0,
                 theta_walk=20.0,
                 theta_sense=20.0,
                 walk_range=1.0,
                 trace_strength=0.3,
                 social_behaviour=-0.5,
                 init_fun=None,
                 template=None,
                 template_strength=1.0,
                 ):
        """Initialize a physarum-population

        :param shape: shape of the simulation-field the population lives in.
        :param horizon_walk: How far the cells walk every-step. Either float or list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param horizon_sense: How far each cell looks out before itself, to check where to go. Either float or a list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param theta_walk: Angle in degrees that the cell turns left or right on every step. Either float or list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param theta_sense: Angle in degrees that the cell checks on the left or right to decide where to go. Either float or a list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param walk_range: Range that will be multiplied with the current horizon_walk value, so that the values for all cells
                        are randomly distributed over it. This allows for faster and slower cells in the same population. Float or tuple (e.g (0.9, 2.1))
        :param trace_strength: Strength of the trace left by the population. This allows to weight different populations differently in their influence. Float, default .3
        :param
        """
        self.shape = shape

        # population is represented as a set of locations (where the organisms are)
        self.population = init_fun(self.shape)
        self.pop_size = len(self.population)

        self.horizon_sense = horizon_sense
        self.social_behaviour = social_behaviour
        self.trace_strength = trace_strength
        self.template_strength = template_strength
        self.walk_range = walk_range

        # as we treat population heading angles in radians, convert the given theta value from degree
        self.angles = np.random.rand(self.pop_size).reshape(-1, 1) * (2 * np.pi)
        self.theta_walk = math.radians(theta_walk)
        self.theta_sense = math.radians(theta_sense)

        self.horizon_walk = horizon_walk
        self.trail_map = np.zeros(self.shape)
        self.template = np.zeros(self.shape) if template is None else np.asarray(template)

    @staticmethod
    def cupy_unique_axis0(array, return_counts=False):
        """
        Support method as cupy currently doesn't support .unique + axis
        :param array:
        :param return_counts:
        :return:
        """
        if len(array.shape) != 2:
            raise ValueError("Input array must be 2D.")

        sortarr = array[np.lexsort(array.T[::-1])]
        mask = np.empty(array.shape[0], dtype=np.bool_)
        mask[0] = True
        mask[1:] = np.any(sortarr[1:] != sortarr[:-1], axis=1)

        if return_counts:
            nonzero = np.nonzero(mask)[0]  # may synchronize
            idx = np.empty((nonzero.size + 1,), nonzero.dtype)
            idx[:-1] = nonzero
            idx[-1] = mask.size
            return sortarr[mask], idx[1:] - idx[:-1]
        else:
            return sortarr[mask]

    def leave_trail(self, additive=False):
        """Each particle leaves it's pheromone trace on the trace array.
        If the trace is additive, it gets added, otherwise the trace array is set to the value of the trace."""

        trace_strength = self.trace_strength
        if additive:
            organisms_pos = np.floor(np.remainder(self.population, np.array(self.trail_map.shape))).astype(int)
            vals, count = Physarum.cupy_unique_axis0(organisms_pos, return_counts=True)
            self.trail_map[vals[:, 0], vals[:, 1]] = self.trail_map[vals[:, 0], vals[:, 1]] + (count * trace_strength)
        else:
            self.trail_map[np.remainder(self.population, self.trail_map.shape)] = trace_strength

        return self.trail_map

    def diffuse_gaussian(self, sigma=7, mode="wrap", truncate=4):
        """Pheromones get distributed using gaussian smoothing."""
        self.trail_map = gaussian_filter(self.trail_map, sigma=sigma, mode=mode, truncate=truncate)

    def diffuse_uniform(self, size=3, mode="wrap"):
        """Pheromones get distributed using uniform smoothing. This can lead to nice artefacts,
        but also to diagonal drift at high values for size (?)"""
        self.trail_map = uniform_filter(self.trail_map, size=size, mode=mode)

    def diffuse_median(self, size=3, mode="wrap"):
        self.trail_map = median_filter(self.trail_map, size=size, mode=mode)

    def update_positions(self, other_populations):
        """Intermediate function, to get everything in order for optimized method"""
        angle = self.angles
        theta_sense =  np.float64(self.theta_sense)
        horizon_sense =  np.float64(self.horizon_sense)
        theta_walk =  np.float64(self.theta_walk)
        horizon_walk =  np.float64(self.horizon_walk)
        population = self.population

        adapted_trail = self.trail_map + (self.template * self.template_strength)

        # include the other species-patterns in the present ones feeding-behaviour
        for this_pop in other_populations:
            adapted_trail = adapted_trail + (this_pop * self.social_behaviour)

        new_population, new_angles = Physarum.optimized_update_positions(population, angle,
                                                                  theta_sense, horizon_sense,
                                                                  theta_walk, horizon_walk,
                                                                  trace_array=adapted_trail)
        self.population = new_population
        self.angles = new_angles

    @staticmethod
    def optimized_update_positions(positions, angle, theta_sense, horizon_sense, theta_walk,
                               horizon_walk, trace_array):
        """Returns the adapted physarum-positions, given initial coordinates and constants.
        This function is optimized by using Cupy (implementation of NumPy-compatible multi-dimensional array on CUDA)"""

        ### Get all possible positions to test
        # get the new 3 angles to test for each organism
        angles_to_test = np.hstack(((angle - theta_sense) % (2 * np.pi),
                                    angle,
                                    (angle + theta_sense) % (2 * np.pi),)).reshape(-1, 3)
        # get positions to test based on current positions and angles
        pos_to_test = positions.reshape(-1, 1, 2) + np.stack((horizon_sense * np.cos(angles_to_test),
                                                              horizon_sense * np.sin(angles_to_test)), axis=-1)
        pos_to_test = np.remainder(pos_to_test, np.array(trace_array.shape))

        ### Get all possible positions to walk to
        # get the new 3 angles to walk to for each organism
        angles_to_walk = np.hstack(((angle - theta_walk) % (2 * np.pi),
                                    angle,
                                    (angle + theta_walk) % (2 * np.pi),)).reshape(-1, 3)
        # get positions to walk to based on current positions and angles
        pos_to_walk = positions.reshape(-1, 1, 2) + np.stack((horizon_walk * np.cos(angles_to_walk),
                                                              horizon_walk * np.sin(angles_to_walk)), axis=-1)
        pos_to_walk = np.remainder(pos_to_walk, np.array(trace_array.shape))

        ### Get the positions to walk too based on the best positions out of the tested ones
        pos_to_test = np.floor(pos_to_test).astype(np.int64) - 1
        # TODO notice argmax will always return first when multiple entries are equal
        best_indexes = trace_array[pos_to_test[:, :, 0], pos_to_test[:, :, 1]].argmax(axis=-1)
        new_positions = pos_to_walk[np.arange(len(pos_to_test)), best_indexes]
        new_angles = angles_to_walk[np.arange(len(pos_to_test)), best_indexes].reshape(-1, 1)

        return new_positions, new_angles


# TODO refactor diffusion and decay process, maybe into the class itself
def run_physarum_simulation(populations, steps, additive_trace=True, diffusion="uniform", mask=None, decay=0.9):
    image_list = []

    for step in tqdm.tqdm_notebook(range(0, steps)):
        species_images = []
        for ix, this_species in enumerate(populations):
            this_species.leave_trail(additive=additive_trace)

            if diffusion == "uniform":
                this_species.diffuse_uniform(size=3)
            elif diffusion == "median":
                this_species.diffuse_median(size=3)
            elif diffusion == "gaussian":
                this_species.diffuse_gaussian(sigma=2, mode="wrap", truncate=5)

            # We can set a predifined mask, e.g. for labyrinths:
            if mask is not None:
                this_species.trail_map[mask] = 0.0

            other_populations = [x for i, x in enumerate(populations) if i != ix]
            other_populations = [species.trail_map for species in other_populations]
            this_species.update_positions(other_populations=other_populations)
            im = this_species.trail_map

            species_images.append(im)
            this_species.trail_map = this_species.trail_map * decay

        im = species_images[0]
        # for image_to_concat in species_images[1:]:
        #    im = IMG.alpha_composite(im, image_to_concat)

        image_list.append(im)
    return np.asnumpy(np.array(image_list))


###############	INIT FUNCTIONS	##################
def leave_feeding_trace(x, y, shape, trace_strength=1.0, sigma=7, mode="wrap", wrap_around=True, truncate=4):
    """Turns x,y-coordinates returned by a list of calls to init-functions into a smooth feeding array."""
    base = np.zeros(shape)
    if wrap_around == True:
        x = x % shape[0]
        y = y % shape[1]
    else:
        print("CutOff overly large x and y: Not yet implemented.")

    base[
        np.floor(x % (base.shape[0])).astype(int),
        np.floor(y % (base.shape[1])).astype(int),
    ] = trace_strength
    base = gaussian_filter(base, sigma=sigma, mode=mode, truncate=truncate)
    return base


def get_perlin_init(shape=(1000, 1000), n=100000, cutoff=None, repetition=(1000, 1000), scale=100, octaves=20.0,
                    persistence=0.1, lacunarity=2.0):

    """Returns a tuple of x,y-coordinates sampled from Perlin noise.
    This can be used to initialize the starting positions of a physarum-
    population, as well as to generate a cloudy feeding-pattern that will
    have a natural feel to it. This function wraps the one from the noise-
    library from Casey Duncan, and is in parts borrowed from here (see also this for a good explanation of the noise-parameters):
    https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401
    The most relevant paramaters for our purposes are:

    :param shape: The shape of the area in which the noise is to be generated. Defaults to (1000,1000)
    :type shape: Tuple of integers with the form (width, height).
    :param n: Number of particles to sample. When used as a feeeding trace,
    this translates to the relative strength of the pattern. defaults to 100000.
    :param cutoff: value below which noise should be set to zero. Default is None. Will lead to probabilities 'contains NaN-error, if to high'
    :param scale: (python-noise parameter) The scale of the noise -- larger or smaller patterns, defaults to 100.
    :param repetition: (python-noise parameter) Tuple that denotes the size of the area in which the noise should repeat itself. Defaults to (1000,1000)

    """
    import numpy as np
    import cupy as cp  # vectorized not present in cupy, so for now to conversion at the end

    shape = [i - 1 for i in shape]

    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, shape[0], shape[0])
    y_idx = np.linspace(0, shape[1], shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(
        world_x / scale,
        world_y / scale,
        octaves=int(octaves),
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=repetition[0],
        repeaty=repetition[1],
        base=np.random.randint(0, 100),
    )
    # world = world * 3
    # 	 Sample particle init from map:
    world[world <= 0.0] = 0.0  # filter negative values
    if cutoff is not None:
        world[world <= cutoff] = 0.0
    linear_idx = np.random.choice(
        world.size, size=n, p=world.ravel() / float(world.sum())
    )
    x, y = np.unravel_index(linear_idx, shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return cp.asarray(np.hstack([x, y]))


def get_circle_init(n: int, center: Tuple[int], radius: int, width: int):
    """Returns tuple of x,y-coordinates sampled from a ring with the given center, radius and width."""
    x = (center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, n))).reshape(-1, 1)
    y = (center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, n))).reshape(-1, 1)
    # perturb coordinates:

    x = x + np.random.normal(0.0, 0.333, size=(n, 1)) * width
    y = y + np.random.normal(0.0, 0.333, size=(n, 1)) * width
    return np.hstack([x, y])


def get_filled_circle_init(n: int, center: Tuple[int], radius: int):
    """Returns tuple of x,y-coordinates sampled from a circle with the given center and radius"""

    t = 2 * np.pi * np.random.rand(n)
    r = np.random.rand(n) + np.random.rand(n)
    r[r > 1] = 2 - r[r > 1]
    x = center[0] + r * radius * np.cos(t)
    y = center[1] + r * radius * np.sin(t)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return np.hstack([x, y])


def get_gaussian_gradient(n: int, center: Tuple[int], sigma: int):
    """Returns tuple of x,y-coordinates sampled from a 2-d gaussian around a given center with a given sigma"""

    x = np.random.normal(center[0], sigma, size=(n, 1))
    y = np.random.normal(center[1], sigma, size=(n, 1))
    return np.hstack([x, y])


def get_uniform_init(n: int, shape: Tuple[int]):
    """Returns tuple of x,y-coordinates uniformly distributed over an area of the given shape."""
    x = np.random.randint(shape[0], size=n).reshape(-1, 1)  # starting_position x
    y = np.random.randint(shape[1], size=n).reshape(-1, 1)  # starting_position y
    return np.hstack([x, y])


def get_image_init_positions(image, shape: Tuple[int], n: int, flip=False):
    init_image = IMG.open(image).convert("L")
    init_image = init_image.resize(tuple(np.flip(shape)))
    init_image = np.array(init_image) / 255
    if flip:
        init_image = 1 - init_image
    linear_idx = np.random.choice(
        init_image.size, size=n, p=init_image.ravel() / float(init_image.sum())
    )
    x, y = np.unravel_index(linear_idx, shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return np.hstack([x, y])


def get_image_init_array(image, shape: Tuple[int]):
    init_image = IMG.open(image).convert("L")
    init_image = init_image.resize(tuple(np.flip(shape)))
    init_image = np.array(init_image) / 255
    return init_image