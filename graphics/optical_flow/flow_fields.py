import numpy as np
from math import cos, sin, radians

# see introductory article https://tylerxhobbs.com/essays/2020/flow-fields

np.random.seed(0)

def compute_streamlines(grid: np.array, nb_lines: int, nb_steps: int, step_size: float):
    all_points = [idx for idx, v in np.ndenumerate(grid)]
    starting_points = np.random.choice(len(all_points), size=nb_lines, replace=False)
    starting_points = [all_points[idx] for idx in starting_points]

    streamlines = []
    for starting_point in starting_points:
        cur_loc_idx = np.array(starting_point)
        cur_loc = cur_loc_idx
        this_line = [cur_loc_idx]
        for step in range(nb_steps):
            cur_angle = grid[cur_loc_idx[0], cur_loc_idx[1], cur_loc_idx[2]]
            next_loc = cur_loc + (np.array([cos(cur_angle), sin(cur_angle), 0]) * step_size)
            if np.any(next_loc < 0) or np.any(next_loc > grid.shape):
                break
            else:
                cur_loc_idx = next_loc.astype(np.int)
                this_line.append(next_loc)
                cur_loc = next_loc

        streamlines.append(this_line)
    return streamlines


grid = np.array(grid)[:, 2].reshape(np.array(grid_shape, dtype=int))
streamlines = compute_streamlines(grid, nb_lines, nb_steps, step_size)