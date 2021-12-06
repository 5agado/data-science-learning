# Adapted from https://github.com/elfnor/hyperbolic_coral/blob/master/hyperbolic_tiling.py


import numpy as np
from math import pi, cos, sin, tan, atan2
from random import choice
import math

ndigits = 5


def distance(p1, p0):
    """euclidean distance """
    d = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
    return d


def point_invert(p: np.array, o: np.array, r: float):
    """
    Get inverse of point P, with respect to a circle centered at O of radius r
    :param p: point to invert
    :param o: center of circle of inversion
    :param r: radius of circle of inversion
    :return:
    """
    if (p == o).all():
        return p
    
    alpha = (r ** 2) / np.sum((p - o) ** 2)

    inverse_of_p = alpha * (p - o) + o
    return inverse_of_p



def circle_invert(p1, p0, r):
    """
    invert p1 in circle centered p0 radius r
    http://users.math.yale.edu/public_html/People/frame/Fractals/
    """
    if r >= 9999.9:
        # reflect in diameter

        a = (p1[0] * p0[0] + p1[1] * p0[1]) / (p0[0] ** 2 + p0[1] ** 2)
        x = 2 * a * p0[0] - p1[0]
        y = 2 * a * p0[1] - p1[1]
    else:
        m = r ** 2 / ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        x = p0[0] + m * (p1[0] - p0[0])
        y = p0[1] + m * (p1[1] - p0[1])

    return x, y


def line_center(p0, p1):
    """
    given two points p0, p1 inside a poincare disk
    find the centre and radius of the arc that defines a line through them
    https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model#Analytic_geometry_constructions_in_the_hyperbolic_plane
    """
    u1, u2, u3 = p0
    v1, v2, v3 = p1

    c = u1 * v2 - u2 * v1

    if abs(c) < 1e-8:
        # points are on diameter
        r = 9999.9
        theta = atan2(p0[1], p0[0])
        a = cos(theta)
        b = sin(theta)
    else:
        a = -0.5 * (u2 * (v1 ** 2 + v2 ** 2) - v2 * (u1 ** 2 + u2 ** 2) + u2 - v2) / c
        b = -0.5 * (v1 * (u1 ** 2 + u2 ** 2) - u1 * (v1 ** 2 + v2 ** 2) + v1 - u1) / c

        r = (a ** 2 + b ** 2 - 1) ** 0.5

    return [a, b, 0], r


def get_circles_intersections(center0, r0, center1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt(np.sum((center1 - center0) ** 2))


    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)

        p0 = center0 + a * (center1 - center0) / d

        x0, y0 = center0[:2]
        x1, y1 = center1[:2]
        x2, y2 = p0[:2]

        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        p1 = np.array([x3, y3, 0])

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d
        p2 = np.array([x4, y4, 0])

        return p1, p2


def reflect_face(edge_ind, face_ind, verts_out, edges_out, faces_out, edge_face_map, verts_rnd):
    """
    reflect face in edge using poincare disk hyperbolic geometry
    returns new unigue vertices, face and edges and updated edge_face_map
    """

    edges_new = []
    verts_new = []

    face = faces_out[face_ind]
    edge = edges_out[edge_ind]

    face_new = [edge[1]]

    # quantize vert_out for detemining if vertex is new
    # verts_rnd = [ [round(v[0], ndigits), round(v[1], ndigits),round(v[2], ndigits) ] for v in verts_out ]

    # find centre and radius of line through edge
    c, r = line_center(verts_out[edge[0]], verts_out[edge[1]])
    # start face at end of edge
    p = len(face)
    ind_start = face.index(edge[1]) - p
    for i in range(p - 1):
        vco = verts_out[face[i + ind_start + 1]]
        vinv = point_invert(np.array(vco), np.array(c), r)
        vinv_rnd = (round(vinv[0], ndigits), round(vinv[1], ndigits), 0.0)

        if vinv_rnd in verts_rnd:
            face_new.append(verts_rnd[vinv_rnd])
        else:
            verts_new.append([vinv[0], vinv[1], 0.0])
            face_new.append(len(verts_out) + len(verts_new) - 1)

    efm = edge_face_map.copy()
    edges_maybe_new = [[v, face_new[i - 1]] for i, v in enumerate(face_new)]

    for edge_new in edges_maybe_new:
        if edge_new in edges_out:
            # edge already exists, delete from edge_face_map
            del (efm[edges_out.index(edge_new)])

        elif edge_new[::-1] in edges_out:
            # edge already exists in opposite order
            del (efm[edges_out.index(edge_new[::-1])])
        else:
            # new edge
            edges_new.append(edge_new)
            key = len(edges_out) + len(edges_new) - 1
            efm[key] = [len(faces_out), min(edge_new)]

    return verts_new, edges_new, face_new, efm


def count_faces(p, q, nlayers):
    """
    find number of faces in poincare regular tiling
    p : number of edges per face
    q : number of faces meeting at a vertex
    nlayers : number of layers or rings of faces
    http://through-the-interface.typepad.com/through_the_interface/2012/01/generating-hyperbolic-tessellations-inside-autocad-using-net.html
    """
    count = 1
    a = p * (q - 3)
    b = p
    for i in range(nlayers - 1):
        count += a + b
        if q == 3:
            next_a = a + b
            next_b = (p - 6) * a + (p - 5) * b
        else:
            next_a = ((p - 2) * (q - 3) - 1) * a + ((p - 3) * (q - 3) - 1) * b
            next_b = (p - 2) * a + (p - 3) * b;
        a = next_a
        b = next_b

    return count


def poke_faces(verts, edges, faces):
    """
    put vertex in the centre of each face and connect it to exising vertices
    """
    verts_out = verts[:]
    edges_out = edges[:]
    faces_out = []

    for f in faces:
        vco = [verts[i] for i in f]
        x = sum([v[0] for v in vco]) / len(f)
        y = sum([v[1] for v in vco]) / len(f)
        z = sum([v[2] for v in vco]) / len(f)
        verts_out.append([x, y, z])

        edges_new = [[len(verts_out) - 1, fi] for fi in f]
        edges_out.extend(edges_new)

        faces_new = [[fi, len(verts_out) - 1, f[i - 1]] for i, fi in enumerate(f)]
        faces_out.extend(faces_new)

    return verts_out, edges_out, faces_out


def fix_normals(verts_out, faces_out):
    # some faces need to be reversed to keep normals in the same direction
    # calculte z component of face normal
    faces_rev = faces_out[:]
    for i, f in enumerate(faces_out):
        a = verts_out[f[0]]
        b = verts_out[f[1]]
        c = verts_out[f[2]]
        t = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if t < 0.0:
            faces_rev[i] = f[::-1]

    return faces_rev


def poincare_tiling(p, q, nlayers, center):
    """
    produce poincare regular tiling
    p : number of edges per face
    q : number of faces meeting at a vertex
    nlayers : number of layers or rings of faces
    """

    max_faces = count_faces(p, q, nlayers)

    edges_out = []
    verts_out = []
    faces_out = []

    # find center inversion
    origin = np.zeros(center.shape)
    center_inverse = point_invert(center, origin, 1)
    midpoint = center + (center_inverse - center)/2
    intersections = get_circles_intersections(origin, 1, midpoint, distance(origin, midpoint))
    
    if intersections is None:
        center = origin
        radius = 1
    else:
        p1, p2 = intersections
        center, radius = line_center(p1, p2)


    # find radius of center face
    cq = cos(pi / q)
    sp = sin(pi / p)

    r = 1 / (((cq * cq) / (sp * sp)) - 1) ** 0.5
    d = 1 / (1 - ((sp * sp) / (cq * cq))) ** 0.5

    # find the intersection of a line and a sphere at (0, d) with radios r
    # could use mathutils.geometry.intersect_line_sphere_2d

    cp = cos(pi / p)

    a = 1
    b = -2 * d * cp
    c = d ** 2 - r ** 2

    x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2.0 * a)
    x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2.0 * a)

    r1 = x2
    # central polygon

    vIDs = range(0, p)
    for i in range(p):
        theta = (2.0 * i + 1.0) * pi / p
        x = r1 * cos(theta)
        y = r1 * sin(theta)
        z = 0.0

        inverted_point = point_invert(np.array([x, y, z]), np.array([center[0], center[1], 0.]), radius)

        verts_out.append(inverted_point)
        edge = [vIDs[i], vIDs[i - 1]]
        edges_out.append(edge)

    faces_out.append(list((vIDs)))
    verts_rnd = {(round(v[0], ndigits), round(v[1], ndigits), round(v[2], ndigits)): i for i, v in enumerate(verts_out)}
    # print(verts_rnd)
    # edge_face_map is a dictionary of the edges adjacent to only one face,
    # where the key is the index of the edge
    # and the item is a list of the face and of the minimum vertex index of the  edge
    edge_face_map = {i: [0, min(edges_out[i])] for i in range(p)}

    while len(faces_out) < max_faces:
        # find the the free edge with the lowest vertex index
        # this works in layers but does not necesarily place faces in angular order
        edge_free = min(edge_face_map, key=lambda k: edge_face_map[k][1])

        # reflect face in this edge
        face = edge_face_map[edge_free][0]

        verts_new, edges_new, face_new, edge_face_map = reflect_face(edge_free, face,
                                                                     verts_out, edges_out, faces_out,
                                                                     edge_face_map, verts_rnd)

        n = len(verts_out)
        vnr = {(round(v[0], ndigits), round(v[1], ndigits), round(v[2], ndigits)): n + i for i, v in
               enumerate(verts_new)}
        verts_rnd.update(vnr)
        
        verts_out.extend(verts_new)
        edges_out.extend(edges_new)
        faces_out.append(face_new)

    faces_out = fix_normals(verts_out, faces_out)

    return verts_out, edges_out, faces_out


# parameters given in the call from animation-nodes
verts_out, edges_out, faces_out = poincare_tiling(p, q, layers, np.array(center))

    
