import numpy as np
import math
import itertools

##############################################################
############## Generators
##############################################################


def get_simplex(n, dist=1):
    """
    An N-simplex is the simplest possible polytope in N-dimensions (formed by N+1 points, all connected).
    In Euclidean space think about the line for 1D, triangle in 2D, tetrahedron in 3D, etc.

    See https://cs.stackexchange.com/questions/69445/how-generate-n-equidistant-points-in-a-n-1-dimensional-space
    for details about construction method

    :param n:
    :param dist:
    :return: points, edged, faces
    """
    # create n-1 equidistant points with fixed independent axis coord=1
    points = np.identity(n)
    # add last point to be equidistant from all the others
    coord = (1 + math.sqrt(n + 1)) / max(1, n)
    points = np.concatenate([points, np.full([1, n], coord)])
    # center at origin
    barycentre = np.sum(points*(1/n), 0)/math.sqrt(2)
    points = points - barycentre
    # scale
    points = np.array(points) * dist

    # add edge for each points pair
    edges = list(itertools.combinations(np.arange(0, n+1), 2))
    # add face for each points triple
    faces = [[int(l) for l in x] for x in itertools.combinations(np.arange(0, n+1), 3)]

    return np.array(points), np.array(edges), faces

def get_hypercube(n, dist=1):
    points = list(itertools.product([-dist/2, dist/2], repeat=n))
    edges = list(itertools.combinations(np.arange(0, len(points)), 2))
    edges = [e for e in edges if np.isclose(np.linalg.norm(np.array(points[e[0]])-np.array(points[e[1]])), dist)]
    faces = _get_object_faces(points, edges)
    return np.array(points), np.array(edges), faces

def get_hyperoctahedron(n, dist=1.):
    points = np.concatenate([np.identity(n), np.identity(n)*-1])
    # scale
    points = np.array(points) * (math.sqrt(2)/2) * dist

    # add edge for each points pair
    edges = list(itertools.combinations(np.arange(len(points)), 2))
    edges = [e for e in edges if np.isclose(np.linalg.norm(np.array(points[e[0]]) - np.array(points[e[1]])), dist)]
    # add face for each points triple
    faces = _get_object_faces(points, edges, nb_vertices=3)

    return np.array(points), np.array(edges), faces


def get_24cell(n, dist=1.):
    points1 = np.array(list(set(itertools.permutations([1, -1]+[0]*(n-2), n))))
    points2 = np.array(list(set(itertools.permutations([1, 1]+[0]*(n-2), n))))
    points3 = np.array(list(set(itertools.permutations([-1, -1]+[0]*(n-2), n))))
    points = np.concatenate([points1, points2, points3])

    # add edge for each points pair
    edges = list(itertools.combinations(np.arange(len(points)), 2))
    edges = [e for e in edges if np.isclose(np.linalg.norm(np.array(points[e[0]]) - np.array(points[e[1]])), math.sqrt(2))]
    # add face for each points triple
    faces = _get_object_faces(points, edges, nb_vertices=3)

    points = np.array(points) * dist
    return np.array(points), np.array(edges), faces


def get_120cell(n, dist=1.):
    raise NotImplementedError


def get_600cell(n, dist=1.):
    raise NotImplementedError


def _get_object_faces(points, edges, nb_vertices=4):
    faces = []
    for p_idx, p in enumerate(points):
        _get_object_faces_rec(edges, [], p_idx, faces, nb_vertices)

    # remove same faces
    faces_set = []
    unique_faces = []
    for face in faces:
        if set(face) not in faces_set:
            unique_faces.append([int(idx) for idx in face])
            faces_set.append(set(face))
    return unique_faces


def _get_object_faces_rec(edges, p_idxs, next_p_idx, faces, nb_vertices):
    if len(p_idxs) >= nb_vertices:
        if p_idxs[0] == next_p_idx:
            if set(p_idxs) not in faces:
                faces.append(p_idxs)
    else:
        if next_p_idx not in p_idxs:
            cur_p_idx = next_p_idx
            for e in edges:
                if cur_p_idx in e:
                    tmp_e = list(e)
                    tmp_e.remove(cur_p_idx)
                    next_p_idx = tmp_e[0]
                    _get_object_faces_rec(edges, p_idxs + [cur_p_idx], next_p_idx, faces, nb_vertices)


##############################################################
############## Projection
##############################################################


def perspective_projection(obj, focal_distance=1., focal_length=1.):
    # TODO was named stereographic_projection when fixed focal-length to 1
    #  while if both focal-distance and focal-length was imaging. What about stereographic projection?
    last_coord = obj[:,-1].reshape(-1, 1) # take last coordinate for each point
    scale = focal_length/((focal_distance - last_coord) + 1.E-7) # define scale based on last-coords and input params
    obj_projection = np.delete(obj, -1, 1) # drop last column. Move from N to N-1 dimensions
    # scale all other coordinates by the scale obtained for each point based on the last coordinate
    obj_projection = obj_projection * scale
    # normalized list coord between 0-1. This info can be used then for visualization purposes (e.g. scaling of points)
    norm_last_coord = (last_coord - last_coord.min())/(last_coord.max() - last_coord.min())
    return obj_projection, norm_last_coord


##############################################################
############## Utils
##############################################################


def rotate(points, angle, axis1, axis2):
    # based on https://stackoverflow.com/questions/45108306/how-to-use-4d-rotors
    # probably not exact formulation, as some signs change depending on axis
    c = math.cos(angle)
    s = math.sin(angle)
    transform_matrix = np.identity(points.shape[-1])
    transform_matrix[axis1, axis1] = c
    transform_matrix[axis1, axis2] = s
    transform_matrix[axis2, axis1] = -s
    transform_matrix[axis2, axis2] = c

    # apply rotation matrix to each point
    for i, p in enumerate(points):
        points[i] = transform_matrix @ p

    return points


def subdivide(points, edges, faces):
    new_points = []
    new_edges = []
    new_edges_map = {}
    for e in edges:
        p0 = points[e[0]]
        p1 = points[e[1]]
        midpoint = p0 + ((p1-p0)/2)
        new_p_idx = len(points)+len(new_points)
        new_points.append(midpoint)
        new_edges_map[frozenset([e[0], e[1]])] = new_p_idx
        new_edges.append((e[0], new_p_idx))
        new_edges.append((new_p_idx, e[1]))
    new_faces = []
    for f in faces:
        new_face = list(itertools.chain(*[[f[f_idx], new_edges_map[frozenset([f[f_idx], f[f_idx+1]])]] for f_idx,_ in enumerate(f[:-1])]))
        new_face = [int(l) for l in new_face]
        new_face.append(int(f[-1]))
        new_face.append(int(new_edges_map[frozenset([f[0], f[-1]])]))
        new_faces.append(new_face)
    return np.append(points, np.array(new_points), 0), np.array(new_edges), new_faces