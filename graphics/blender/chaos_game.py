import numpy as np
from math import pi, cos, sin


def get_polygon(center: tuple, radius: float, segments: int):
    """Util to get a polygon points"""
    polygon = []
    angle = 2*pi/segments  # angle in radians
    for i in range(segments):
        x = center[0] + radius*cos(angle*i)
        y = center[1] + radius*sin(angle*i)
        z = center[2]
        polygon.append([x, y, z])

    return polygon


def _get_polygon_center(polygon: np.array):
    centroid = polygon.mean(axis=0)
    return centroid

def _get_midpoint(p0, p1, dist_factor):
    return p0 + ((p1 - p0) * dist_factor)

def run_chaos_game(polygon, nb_iter: int, dist_factor=0.5, vertex_choice_constrain=''):
    """
    Chaos game: start from a point inside a polygon and iteratively move to a point between the current position and a randomly chosen vertex of the polygon.
    Constraints around distance-factor and vertex-choice can be added to obtain different fractal structures.
    :param polygon: list of points of the input polygon
    :param nb_iter: number of iterations to run the game
    :param dist_factor: factor used to chose the next point along the line between the current point and the randomly chosen vertex
    :param vertex_choice_constrain: additional constrain posed on the polygon vertex selection. Current option: '', 'skip_last', 'skip_last_neighbors'
    :return:
    """
    # get starting point
    # TODO currently starting from the polygon center
    p_current = _get_polygon_center(np.array(polygon))
    all_points = [p_current]
    prev_polygon_idx = -1

    for i in range(int(nb_iter)):
        # get a random vertex from the current polygon
        if vertex_choice_constrain == 'skip_last':
            possible_polygon_idxs = [idx for idx in range(len(polygon)) if idx != prev_polygon_idx]
        elif vertex_choice_constrain == 'skip_last_neighbors':
            possible_polygon_idxs = [idx for idx in range(len(polygon)) if abs(idx - prev_polygon_idx) > 1]
        elif vertex_choice_constrain == '':
            possible_polygon_idxs = range(len(polygon))
        else:
            raise Exception(f'No such vertex_choice_constrain: {vertex_choice_constrain}')
        prev_polygon_idx = np.random.choice(possible_polygon_idxs)
        rand_p = polygon[prev_polygon_idx]
        # compute the midpoint between current point and the random polygon vertex
        p_next = _get_midpoint(p_current, rand_p, dist_factor)
        p_next[2] = np.sin(i/100)
        # append the new point and make it the current one
        all_points.append(p_next)
        p_current = p_next

    return all_points

# To run in Blender animation-nodes
#np.random.seed(seed)
#all_points = run_chaos_game(np.array(polygon), nb_iter=nb_iter, dist_factor=dist_factor)