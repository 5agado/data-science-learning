import argparse
import sys
import subprocess
from pathlib import Path
import math
from PIL import Image, ImageEnhance

from ds_utils.image_processing import get_imgs_paths

# relies on [ImageMagick](http://www.imagemagick.org/Usage/montage/)


def main(_=None):
    parser = argparse.ArgumentParser(description='In-place Image pre-process via ImageMagick')
    parser.add_argument('-i', '--inp', required=True, help="input folder containing the images")

    # Parse args
    args = parser.parse_args()
    input_dir = Path(args.inp)

    image_paths = get_imgs_paths(input_dir, img_types=('*.jpg', '*.jpeg', '*.png', '*.webp'))

    for i, p in enumerate(image_paths):
       command = f'magick mogrify -resize 1600^> "{p}"'  # -format png
       subprocess.call(command, shell=True)

       img = Image.open(p)
       img = img.convert('RGB')
       converter = ImageEnhance.Color(img)
       edited_img = converter.enhance(1.3)
       #edited_img = ImageEnhance.Contrast(edited_img).enhance(2)
       edited_img.save(p)


    #for i, p in enumerate(image_paths):
    #    command = f'magick {p} -gravity center -extent "%[fx:h<w?h:w]x%[fx:h<w?h:w]" {input_dir}/cropped/{i}_.png'
    #    subprocess.call(command, shell=True)
    #return


if __name__ == "__main__":
    main(sys.argv[1:])
