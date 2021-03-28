import argparse
import sys
import subprocess
from pathlib import Path
import math

from ds_utils.image_processing import get_imgs_paths

# relies on [ImageMagick](http://www.imagemagick.org/Usage/montage/)

def main(_=None):
    parser = argparse.ArgumentParser(description='Video Conversion via FFMPEG')
    parser.add_argument('-i', '--inp', required=True, help="input path")
    parser.add_argument('-o', '--out', required=True, help="output path")

    # Parse args
    args = parser.parse_args()
    input_dir = Path(args.inp)

    image_paths = get_imgs_paths(input_dir)

    # Generate commands parts
    input_line = " ".join([f"{path}" for path in image_paths])

    # Run command
    command = f'magick convert ' \
              f'{input_line} ' \
              f'-flatten ' \
              f'{args.out}'

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
