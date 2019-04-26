import argparse
import sys
import subprocess
from pathlib import Path
import math

from ds_utils.image_processing import get_imgs_paths

# relies on [ImageMagick](http://www.imagemagick.org/Usage/montage/)


def main(_=None):
    parser = argparse.ArgumentParser(description='Image Montage via ImageMagick')
    parser.add_argument('-i', '--inp', required=True, help="input folder containing the images")
    parser.add_argument('-o', '--out', required=True, help="output path")
    parser.add_argument('-r', '--rows', default=0, help="number of rows in the montage")
    parser.add_argument('-c', '--cols', default=0, help="number of columns in the montage")

    # Parse args
    args = parser.parse_args()
    input_dir = Path(args.inp)
    nb_rows = int(args.rows)
    nb_cols = int(args.cols)

    # Validate mosaic size against number of videos
    image_paths = get_imgs_paths(input_dir)
    nb_images = len(image_paths)
    assert (nb_cols*nb_cols) <= nb_images

    if nb_cols == 0 and nb_rows == 0:
        nb_cols = nb_rows = int(math.sqrt(nb_images))
    else:
        if nb_cols == 0:
            nb_cols = int(nb_images/nb_rows)
        if nb_rows == 0:
            nb_rows = int(nb_images/nb_cols)

    image_paths = image_paths[:(nb_cols * nb_cols)]

    # Generate commands parts
    input_line = " ".join([f"{path}" for path in image_paths])

    # Run command
    command = f'montage '\
              f'{input_line} '\
              f'-geometry +{nb_rows}+{nb_cols} '\
              f'{args.out}'

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main(sys.argv[1:])
