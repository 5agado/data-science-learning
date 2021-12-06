import numpy as np

def project_point(p, a, b):
    """
    Project point p onto segment ab
    :param p:
    :param a:
    :param b:
    :return:
    """
    ap = p - a
    ab = b - a
    t = np.dot(ap,ab)/np.dot(ab,ab)
    t = max(0, min(1, t))
    res = a + t * ab
    return res


def get_polygon_vertices_ordered(polygons_vertices, polygons_indices, edge_indices):
    """
    Reorder polygon vertices such that they are connected (Blender order can sometimes skip around)
    :param polygons_vertices:
    :param polygons_indices:
    :param edge_indices:
    :return:
    """
    p0_idx, p1_idx, p2_idx, p3_idx =polygons_indices
    p0 = np.array(polygons_vertices[p0_idx])

    if (p0_idx, p1_idx) in edge_indices or (p1_idx, p0_idx) in edge_indices:
        p1 = np.array(polygons_vertices[p1_idx])
        if (p1_idx, p2_idx) in edge_indices or (p2_idx, p1_idx) in edge_indices:
            p2 = np.array(polygons_vertices[p2_idx])
            p3 = np.array(polygons_vertices[p3_idx])
        else:
            p2 = np.array(polygons_vertices[p3_idx])
            p3 = np.array(polygons_vertices[p2_idx])
    else:
        p1 = np.array(polygons_vertices[p2_idx])
        p2 = np.array(polygons_vertices[p1_idx])
        p3 = np.array(polygons_vertices[p3_idx])

    return [p0, p1, p2, p3]

def subdivide_polygon(polygon, normal, a_ratio, b_ratio, c_ratio, z_ratio):
    """
    Main subdivision function
    :param polygon: quad polygon to subdivide
    :param normal: polygon normal
    :param a_ratio: ratio to define where a sits between p0 and p1
    :param b_ratio: ratio to define where b sits between p1 and p2
    :param c_ratio: ratio to define where c sits between p2 and p3
    :param z_ratio: ratio for random normal z displacement
    :return:
    """
    p0, p1, p2, p3 = polygon

    a_u = np.random.uniform(0, 1) if a_ratio == 0. else a_ratio
    b_u = np.random.uniform(0, 1) if b_ratio == 0. else b_ratio
    c_u = np.random.uniform(0, 1) if c_ratio == 0. else c_ratio

    a = a_u*p0 + (1-a_u)*p1
    b = b_u*p1 + (1-b_u)*p2
    c = c_u*p2 + (1-c_u)*p3
    d = project_point(b, p3, p0) 

    e = project_point(a, b, d) 
    f = project_point(c, b, d) 
    
    z_ratio = z_ratio*5
    pol01 = np.array([p0, a, e, d]) + normal * ((np.random.rand()-0.5)/z_ratio)
    pol02 = np.array([a, p1, b, e]) + normal * ((np.random.rand()-0.5)/z_ratio)
    pol03 = np.array([f, b, p2, c]) + normal * ((np.random.rand()-0.5)/z_ratio)
    pol04 = np.array([d, f, c, p3]) + normal * ((np.random.rand()-0.5)/z_ratio)

    return [pol01, pol02, pol03, pol04]

def rec_subdivide(polygons, normals, a_ratio, b_ratio, c_ratio, z_ratio, cur_iter, max_iters):
    """
    Recursive subdivision function
    :param cur_iter: current iteration number
    :param max_iters: maximum number of iterations
    :return:
    """
    if cur_iter >= max_iters:
        return polygons
    else:
        new_polygons = []
        new_normals = []
        for i, p in enumerate(polygons):
            p_normal = np.array(normals[i])
            new_polygons.extend(subdivide_polygon(p, p_normal, a_ratio, b_ratio, c_ratio, z_ratio))
            new_normals.extend([p_normal]*4)
        return rec_subdivide(new_polygons, new_normals, a_ratio, b_ratio, c_ratio, z_ratio, cur_iter+1, max_iters)

np.random.seed(seed)
polygons_in = [get_polygon_vertices_ordered(list(polygons_vertices), p, list(edges_indices)) for p in list(polygons_indices)]
polygons = rec_subdivide(polygons_in, normals, a_ratio, b_ratio, c_ratio, z_ratio, 0, max_iters)