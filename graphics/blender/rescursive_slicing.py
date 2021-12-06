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

def scale_polygon(p, scale_factor):
    """
    Scale polygon towards its center
    :param p:
    :param scale_factor:
    :return:
    """
    p = np.array(p)
    center = np.mean(p, axis=0)
    vecs_to_center = p - center
    scaled_p = p - (scale_factor * vecs_to_center)
    return scaled_p

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

def slice_polygon(polygon, normal, a_ratio, b_ratio, z_ratio):
    """
    Main slicing function
    :param polygon: quad polygon to slice
    :param normal: polygon normal
    :param a_ratio: ratio to define where a sits between p0 and p1
    :param b_ratio: ratio to define where b sits between p1 and p2
    :param c_ratio: ratio to define where c sits between p2 and p3
    :param z_ratio: ratio for random normal z displacement
    :return:
    """
    # shift polygon to randomize slicing orientation
    polygon = np.roll(np.array(polygon), np.random.randint(len(polygon)), axis=0)
    p0, p1, p2, p3 = polygon

    a_u = np.clip(np.random.normal(0.5, a_ratio), 0.1, 0.9) #if a_ratio == 0. else a_ratio
    b_u = np.clip(np.random.uniform(0.5, b_ratio), 0.1, 0.9) #if b_ratio == 0. else b_ratio

    a = a_u*p0 + (1-a_u)*p1
    b = b_u*p2 + (1-b_u)*p3
    #b = project_point(a, p2, p3)

    z_ratio = z_ratio*5
    pol01 = np.array([p0, a, b, p3]) + normal * ((np.random.rand()-0.5)/z_ratio)
    pol02 = np.array([a, p1, p2, b]) + normal * ((np.random.rand()-0.5)/z_ratio)

    return [pol01, pol02]

def rec_slicing(polygons, normals, a_ratio, b_ratio, slice_likelihood, scale_factor, z_ratio, cur_iter, max_iters):
    """
    Recursive slicing function
    :param cur_iter: current iteration number
    :param max_iters: maximum number of iterations
    :return:
    """
    if cur_iter >= max_iters:
        return polygons
    else:
        old_polygons = []
        new_polygons = []
        new_normals = []
        for i, p in enumerate(polygons):
            p = scale_polygon(p, scale_factor=scale_factor)
            p_normal = np.array(normals[i])
            # randomly decide whether to slice a polygon or not
            if cur_iter < 2 or np.random.rand() >= slice_likelihood:
                new_polygons.extend(slice_polygon(p, p_normal, a_ratio, b_ratio, z_ratio))
                new_normals.extend([p_normal]*4)
            else:
                old_polygons.extend([p])
        return rec_slicing(new_polygons, new_normals, a_ratio, b_ratio, slice_likelihood, scale_factor, z_ratio, cur_iter+1, max_iters) + old_polygons

np.random.seed(seed)
polygons_in = [get_polygon_vertices_ordered(list(polygons_vertices), p, list(edges_indices)) for p in list(polygons_indices)]
polygons = rec_slicing(polygons_in, normals, abs(a_ratio), abs(b_ratio), slice_likelihood, scale_factor, z_ratio, 0, max_iters)