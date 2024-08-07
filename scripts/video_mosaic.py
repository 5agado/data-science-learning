import argparse
import sys
import os
import subprocess
from pathlib import Path

from ds_utils.image_processing import get_imgs_paths

# see https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos


def main(_=None):
    parser = argparse.ArgumentParser(description='Video Mosaic via FFMPEG')
    parser.add_argument('-i', '--inp', required=True, help="input folder containing the videos")
    parser.add_argument('-o', '--out', required=True, help="output path")
    parser.add_argument('--width', required=True, help="width of a single video cell")
    parser.add_argument('--height', required=True, help="height of a single video cell")
    parser.add_argument('-r', '--rows', required=True, help="number of rows in the mosaic")
    parser.add_argument('-c', '--cols', required=True, help="number of columns in the mosaic")
    parser.add_argument('--time-sort', action='store_true', help='Sort videos by created time')
    parser.set_defaults(time_srt=False)
    parser.add_argument('--shortest', action='store_true', help='Cut at shortest, otherwise is repeat last frame')
    parser.set_defaults(shortest=False)

    # Parse args
    args = parser.parse_args()
    input_dir = Path(args.inp)
    width = int(args.width)
    height = int(args.height)
    nb_rows = int(args.rows)
    nb_cols = int(args.cols)
    time_sort = args.time_sort
    shortest = args.shortest

    # Validate mosaic size against number of videos
    video_paths = get_imgs_paths(input_dir, img_types=('*.mp4', '*.mkv', '*.gif'),
                                 sort_by=os.path.getmtime if time_sort else None)[:(nb_rows*nb_cols)]
    nb_videos = len(video_paths)
    assert (nb_cols*nb_rows) <= nb_videos

    overlay_option = 'shortest' if shortest else 'repeatlast'

    # Generate commands parts
    input_line = " ".join([f'-i "{path}"' for path in video_paths])
    entries_line = "; ".join([f"[{i}:v] setpts=PTS-STARTPTS, scale={width}x{height} [vid{i}]"
                              for i in range(nb_videos)])
    pos_line = "; ".join([f"[tmp{nb_cols*row+col}][vid{nb_cols*row+col}] "
                          f"overlay={overlay_option}=1:x={width*col}:y={height*row} [tmp{nb_cols*row+col+1}]"
                          for col in range(nb_cols) for row in range(nb_rows)])[:-7]
    pos_line = f"[base][vid0] overlay={overlay_option}=1:x=0:y=0 [tmp1]; " + pos_line.split(';', 1)[-1]

    # Run command
    command = f'ffmpeg '\
              f'{input_line} '\
              f'-filter_complex "nullsrc=size={nb_cols*width}x{nb_rows*height} [base]; '\
              f'{entries_line}; '\
              f'{pos_line}" '\
              f'-c:v libx264 {args.out}'

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
