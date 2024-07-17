import argparse
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
import numpy as np
import cv2
from ast import literal_eval

sys.path.append("../")
from ds_utils import image_processing
from ds_utils import video_utils

from face_utils import CONFIG_PATH
from face_utils.face_extract_utils import get_face_mask
from face_utils.FaceDetector import FaceDetector, FaceExtractException


def face_mask_fun(frame, frame_count, face_detector: FaceDetector, output_path: Path,
                  mask_type='hull'):
    try:
        faces = face_detector.detect_faces(frame, min_width=face_detector.config['extract']['min_width'])
        mask = np.zeros(frame.shape, np.uint8)

        for face in faces:
            rect = face.rect
            cv2.rectangle(mask, (rect.left, rect.top), (rect.right, rect.bottom), (255, 255, 255), -1)
            #mask = get_face_mask(face, mask_type=mask_type, blur_size=15)

        cv2.imwrite(str(output_path), mask)
        frame_count += 1
    except FaceExtractException as e:
        logging.debug(f"Frame {frame_count}: {e}")
        mask = np.ones(frame.shape, np.uint8)
        cv2.imwrite(str(output_path), mask)
        frame_count += 1
    except Exception as e:
        logging.error(e)
        raise


def frame_extract_fun(frame, frame_count, face_detector, output_path: Path, step_mod: int):
    try:
        faces = face_detector.detect_faces(frame, min_width=face_detector.config['extract']['min_width'])
        for face_count, face in enumerate(faces):
            extracted_face = face_detector.extract_face(face)

            if frame_count % step_mod == 0:
                cv2.imwrite(str(output_path / "face_{:04d}_{:04d}.jpg".format(frame_count, face_count)),
                            extracted_face)
                frame_count += 1
    except FaceExtractException as e:
        logging.debug(f"Frame {frame_count}: {e}")
    except Exception as e:
        logging.error(e)
        raise


def extract_faces(input_path: Path, output_path: Path, config_path: Path, process_images: bool,
                  step_mod: int, mask_type=None, crop_video = False):
    assert input_path.exists(), f"No such path: {input_path}"
    assert config_path.exists(), f"No such config file: {config_path}"

    if not output_path.exists():
        logging.info(f"Creating output dir: {output_path}")
        output_path.mkdir()

    with open(str(config_path), 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)

    face_detector = FaceDetector(cfg)
    frame_count = 0

    if process_images:
        # collected all image paths
        img_paths = image_processing.get_imgs_paths(input_path, as_str=False)

        logging.info("Running Face Extraction over images")
        # iterate over all collected image paths
        for img_path in tqdm(img_paths):
            frame_count += 1
            img = cv2.imread(str(img_path))
            if mask_type is not None:
                face_mask_fun(img, frame_count, face_detector, output_path / f"{img_path.stem}.png",
                              mask_type=mask_type)
            else:
                frame_extract_fun(img, frame_count, face_detector, output_path, step_mod)
    # process video
    else:
        # get a valid file from given directory
        if input_path.is_dir():
            video_files = image_processing.get_imgs_paths(input_path, img_types=('*.gif', '*.webm', '*.mp4'),
                                                          as_str=True)
        else:
            video_files = [input_path]

        if not video_files:
            logging.error(f"No valid video files in: {input_path}")
            sys.exit(1)
        # for now just pick first one
        if not crop_video:
            input_path = Path(video_files[0])

            logging.info("Running Face Extraction over video")

            video_utils.process_video(str(input_path), lambda frame, frame_count:
                                      frame_extract_fun(frame, frame_count, face_detector, output_path, step_mod))
        else:
            for input_video in video_files:
                crop_face_from_video(input_video, str(output_path / f"{Path(input_video).stem}.mp4"), face_detector)


def crop_face_from_video(video_path, out_path, face_detector, fps=None):
    """
    Crop video based on first detected face.
    :param video_path:
    :param out_path:
    :param face_detector:
    :param fps:
    :return:
    """
    faces = []

    # get first frame
    input_video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = input_video.read()
        input_video.release()

        if not ret:
            break

        # extract face from first frame
        try:
            faces = face_detector.detect_faces(frame)
            break
        except FaceExtractException as _:
            continue
        except Exception as e:
            print(frame.shape)
            print(e)
            raise
    input_video.release()

    if not faces:
        print(f'No faces found for {video_path}')
        return

    f_rect = faces[0].rect

    # border expand
    border_expand = literal_eval(face_detector.config['extract']['border_expand'])
    border_expand = (int(border_expand[0] * f_rect.get_size()[0]), int(border_expand[1] * f_rect.get_size()[1]))
    f_rect = f_rect.resize(border_expand)

    crop_frame = lambda frame: frame[f_rect.top:f_rect.bottom, f_rect.left:f_rect.right]
    cropped_frame = crop_frame(frame)

    video_utils.convert_video_to_video(video_path, out_path, crop_frame,
                                       codec='mp4v', is_color=True, fps=fps,
                                       new_size=cropped_frame.shape[:2][::-1])

def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Extract Faces')

    parser.add_argument('-i', '--input-path', required=True)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('-c', '--config-path', default=CONFIG_PATH)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--process-images', action='store_true',
                        help="Run extraction on images in the given input dir")
    parser.set_defaults(process_images=False)
    parser.add_argument('-m', '--mask-type', default=None,
                        help="Output masked results, with the given type of mask (hull or landmarks)")
    parser.set_defaults(extract_mask=False)
    parser.add_argument('-s', '--step-mod', default=1,
                        help="Save only face for frame where frame_num%step_mod == 0")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    process_images = args.process_images
    mask_type = args.mask_type
    config_path = Path(args.config_path)
    step_mod = int(args.step_mod)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extract_faces(input_path, output_path, config_path, process_images, step_mod, mask_type=mask_type,
                  crop_video=False)


if __name__ == "__main__":
    main(sys.argv[1:])
