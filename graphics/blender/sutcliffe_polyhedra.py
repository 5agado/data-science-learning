import mathutils

opposite_point_init = opposite_point
dist_ratio = dist_ratio

def _get_polygon_center(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    z = [p[2] for p in polygon]
    centroid = (sum(x) / len(polygon), sum(y) / len(polygon), sum(z) / len(polygon))
    return centroid

def _get_midpoint(p0, p1):
    return (p0[0] + p1[0]) / dist_ratio, (p0[1] + p1[1]) / dist_ratio, (p0[2] + p1[2]) / dist_ratio


def polygon_recursive(polygon, step=0, max_steps=3, opposite_point_init=(0, 0, 0), all_polygons=None):
    all_polygons.extend([Vector(tuple(v)) for v in polygon] + [Vector(tuple(polygon[-1]))])
    if step >= max_steps:
        return
    else:
        new_polygon = []
        midpoints = []
        for i in range(len(polygon)):
            p0 = polygon[i]
            new_point = _get_midpoint(p0, opposite_point_init)
            new_polygon.append(new_point)
            midpoints.append(p0)
        for i in range(len(polygon)):
            other_polygon = [polygon[i], midpoints[i - 1], new_polygon[i - 1], new_polygon[i], midpoints[i]]
            new_opposite_point = _get_midpoint(_get_polygon_center(other_polygon), _get_polygon_center(polygon))
            polygon_recursive(other_polygon, step + 1, max_steps, new_opposite_point, all_polygons)


# print(polygon)
all_polygons = []
polygon_recursive(polygon, step=0, max_steps=max_steps, opposite_point_init=opposite_point_init,
                  all_polygons=all_polygons)
return all_polygons