import numpy as np
# https://github.com/dranjan/python-plyfile
from plyfile import PlyData, PlyElement


def prepare_for_ply(norm_snapshots, val_threshold):
    t, x, y, z = norm_snapshots.shape
    T, X, Y, Z = np.mgrid[:t, :x, :y, :z]

    pad = np.zeros(len(X.ravel()), dtype=np.uint8)
    out = np.column_stack((T.ravel(), X.ravel(), Y.ravel(), Z.ravel(), pad, pad, norm_snapshots.ravel())).astype(np.uint16)

    out = out.reshape(len(norm_snapshots), -1, 7)
    # out = out[out[:, :,-1] >= val_threshold]
    out = np.delete(out, 0, 2)
    return out


def write_to_ply(in_array, out_filepath):
    vertex = np.array(list(map(tuple, in_array)),
                      dtype=[
                          ('x', 'i2'), ('y', 'i2'), ('z', 'i2'),
                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                      ])

    el = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['points']
            ),
        ]
    )

    # faster and smaller size with "text" false
    PlyData(el, text=False).write(out_filepath)


# http://www.open3d.org/docs/release/index.html
import open3d as o3d

def write_to_ply_o3d(in_array, out_filepath, pcd=None):
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(in_array[0, :, :3])
    pcd.colors = o3d.utility.Vector3dVector(in_array[0, :, 3:])
    o3d.io.write_point_cloud(str(out_filepath), pcd)