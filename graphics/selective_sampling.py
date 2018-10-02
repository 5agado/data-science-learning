import numpy as np
import argparse
import logging
from tqdm import tqdm
import cv2
import sys
from pathlib import Path

# data-science-utils
from utils import image_processing


def selective_sampling(img: np.array, nb_width_cells: int,
                       sampling_size_factor: float= 1/5,
                       force_square_sample=False):
    """
    Samples rectangles from the image at intersection points of a NxM grid where N=nb_width_cells, and M depends on
    the given parameters.
    :param img: image to sample from
    :param nb_width_cells: number of cells for the grid width
    :param sampling_size_factor: multiplicative factor used to define the sample rectangle size.
    Used against the resulting grid cells size
    :param force_square_sample: for the sample to be a square. If False M=N, otherwise M depends on the sample width.
    :return: reconstructed sample
    """
    img_height, img_width = img.shape[:2]

    # Compute grid intersection point along image width
    x_steps, x_step_size = np.linspace(0, img_width, nb_width_cells, retstep=True)
    # same for image height
    # if forced square, just rely on the x_step_size
    if force_square_sample:
        y_steps = np.arange(0, img_height, x_step_size)
        y_step_size = x_step_size
    # otherwise compute independently
    else:
        y_steps, y_step_size = np.linspace(0, img_height, nb_width_cells, retstep=True)

    samples = []
    sample_width = x_step_size * sampling_size_factor
    sample_height = y_step_size * sampling_size_factor
    # Consider all intersection which are not on the image border
    for y in y_steps[1:-1]:
        samples_row = []
        for x in x_steps[1:-1]:
            # Get sample square of size sample_side surrounding the intersection of of current x and y steps
            sample = img[int(y - sample_height):int(y + sample_height),
                         int(x - sample_width):int(x + sample_width)]
            samples_row.append(sample)
        # Concatenate row samples to form unique row (so we need to use axis=1)
        samples.append(np.concatenate(samples_row, axis=1))
    # Concatenate all rows
    result_img = np.concatenate(samples, axis=0)

    return result_img


def convert_video(video_path: str, out_path: str, frame_edit_fun,
                  codec='mp4v'):
    # "Load" input video
    input_video = cv2.VideoCapture(video_path)

    # Match source video features.
    fps = input_video.get(cv2.CAP_PROP_FPS)
    size = (
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, size)

    # Process frame by frame

    # some codecs don't support this, so in such cases we need to rollback to base looping
    nb_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if nb_frames and nb_frames > 0:
        for _ in tqdm(range(nb_frames)):
            _, frame = input_video.read()

            frame = frame_edit_fun(frame)
            out.write(frame)
    else:
        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret:
                frame = frame_edit_fun(frame)
                out.write(frame)
            else:
                break

    # Release everything if job is finished
    input_video.release()
    out.release()


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Deep-Faceswap. Generative based method')

    parser.add_argument('-i', metavar='input_path', dest='input_path', required=True)
    parser.add_argument('-o', metavar='output_path', dest='output_path', required=True)
    parser.add_argument('--process_images', dest='process_images', action='store_true')
    parser.set_defaults(process_images=False)
    # sampling params
    parser.add_argument('-square', dest='force_square_sample', action='store_true')
    parser.set_defaults(force_square_sample=False)
    parser.add_argument('-w', dest='nb_width_cells', default=10)
    parser.add_argument('-f', dest='sampling_size_factor', default=1/5)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    process_images = args.process_images
    force_square_sample = args.force_square_sample
    nb_width_cells = int(args.nb_width_cells)
    sampling_size_factor = float(args.sampling_size_factor)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # process directly a video
    if not process_images:
        logging.info("Running selective sampling over video")
        try:
            frame_edit_fun = lambda x: selective_sampling(x, nb_width_cells, sampling_size_factor, force_square_sample)
            convert_video(str(input_path), str(output_path), frame_edit_fun)
        except Exception as e:
            logging.error(e)
            raise
    # or process a list of images
    else:
        # collected all image paths
        img_paths = image_processing.get_imgs_paths(input_path, as_str=False)

        # iterate over all collected image paths
        logging.info("Running selective sampling over images")
        for img_path in tqdm(img_paths):
            from_filename = img_path.name
            res_path = str(output_path / 'sampled_{}.png'.format(from_filename.split('.')[0]))
            try:
                from_img = cv2.imread(str(input_path / from_filename))
                results = selective_sampling(from_img, nb_width_cells, sampling_size_factor, force_square_sample)
                cv2.imwrite(res_path, results)
            except Exception as e:
                logging.error(e)
                raise


if __name__ == "__main__":
    main(sys.argv[1:])
