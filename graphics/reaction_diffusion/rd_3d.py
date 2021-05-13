import sys
import cv2
import numpy as np
from collections import namedtuple
from itertools import starmap, product
from pathlib import Path

from ReactionDiffusionSystem import ReactionDiffusionSystem, get_init_state, get_cube_mask, ReactionDiffusionException
from ReactionDiffusionSystem import SYSTEM_CORAL_CONFIG, SYSTEM_BACTERIA_CONFIG, SYSTEM_SPIRALS_CONFIG, SYSTEM_ZEBRA_CONFIG
from ds_utils.video_utils import generate_video


def get_grid_search_configs(nb_vals=2) -> list:
    def named_configs(items):
        Config = namedtuple('Config', items.keys())
        return starmap(Config, product(*items.values()))

    grid_search_params = {
        'COEFF_A': np.linspace(0.12, 0.16, nb_vals),
        'COEFF_B': np.linspace(0.06, 0.08, nb_vals),
        'FEED_RATE': np.linspace(0.02, 0.0625, nb_vals),
        'KILL_RATE': np.linspace(0.05, 0.066, nb_vals),
    }
    configs = list(named_configs(grid_search_params))
    return configs


def frame_gen_3d(frame_count, z_coord, rf_snapshots):
    img = cv2.normalize(rf_snapshots[frame_count][z_coord], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img

def main(_=None):
    NUM_FRAMES = 240
    min_frames = 100

    # system config
    size = 100
    nb_configs_gs_param = 2
    config = SYSTEM_ZEBRA_CONFIG.copy()
    config['steps'] = 30
    config['random_influence'] = 0.
    config['validate_change_threshold'] = 1.e-6
    system_shape = tuple([size]*3)

    out_path = Path.home() / f'Documents/graphics/generative_art_output/reaction_diffusion/3d_slicing/{size}x{size}'
    out_path.mkdir(exist_ok=False, parents=True)

    # mask
    mask = get_cube_mask(system_shape, system_shape[0]//5, np.array(system_shape) // 2)
    system_init_fun = lambda shape: get_init_state(shape, random_influence=config['random_influence'], mask=mask)

    configs = get_grid_search_configs(nb_configs_gs_param)
    with open(str(out_path / 'logs.txt'), 'w+') as f:
        for run, param_config in enumerate(configs):
            print(f'#####################')
            print(f'Run {run}')
            rf_snapshots = []
            config.update(param_config._asdict())

            # init reaction diffusion system
            rf_system = ReactionDiffusionSystem(system_shape, config, system_init_fun,
                                                validate_change_threshold=config['validate_change_threshold'])

            # run and store snapshot
            for i in range(NUM_FRAMES):
                try:
                    rf_system.run_simulation(config['steps'])
                    rf_snapshots.append(rf_system.B)
                except ReactionDiffusionException as e:
                    print(f'System throw exception at frame {i} {e}')
                    break
                if i % 50 == 0:
                    print('Frame ', i)

            # write out config
            config['nb_frames'] = len(rf_snapshots)
            f.write(str(config) + '\n')

            # discard if too short
            if len(rf_snapshots) < min_frames:
                continue

            # write out numpy 4D tensor
            np.save(out_path / f'run_{run}.npy', np.array(rf_snapshots, dtype=np.float16))

            # write out as sliced videos
            run_out_path = out_path / f'vid_run_{run:03}'
            run_out_path.mkdir(exist_ok=False, parents=True)
            for z_coord in range(system_shape[0]):
                generate_video(str(run_out_path / f"{z_coord}.mp4"),
                               (rf_system.shape[2], rf_system.shape[1]),
                               frame_gen_fun=lambda i: frame_gen_3d(i, z_coord, rf_snapshots),
                               nb_frames=len(rf_snapshots), is_color=False, disable_tqdm=True)


if __name__ == "__main__":
    main(sys.argv[1:])