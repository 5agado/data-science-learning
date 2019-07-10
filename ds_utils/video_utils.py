import numpy as np
import cv2
from typing import Tuple
from tqdm import tqdm

# Option of relying on MoviePy (http://zulko.github.io/moviepy/index.html)


def generate_video(out_path: str, shape: Tuple[int], frame_gen_fun, nb_frames: int,
                   codec='mp4v', fps=24, is_color=False):
    """
    Write generated frames to file
    :param out_path:
    :param shape:
    :param frame_gen_fun: function that given a frame_count returns an image of the given shape
    :param nb_frames:
    :param codec: default mp4v
    :param fps: default 24
    :param is_color: default False
    :return:
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, shape, is_color)

    for frame_count in tqdm(range(nb_frames)):
        frame = frame_gen_fun(frame_count)
        out.write(frame)

    # Release everything if job is finished
    out.release()


def process_video(input_video_path: str, frame_fun):
    """
    Load video and apply the given function to each frame.
    :param input_video_path:
    :param frame_fun:
    """
    # "Load" input video
    input_video = cv2.VideoCapture(input_video_path)

    # Process frame by frame

    # some codecs don't support this, so in such cases we need to rollback to base looping
    nb_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if nb_frames and nb_frames > 0:
        for frame_count in tqdm(range(nb_frames)):
            _, frame = input_video.read()
            frame_fun(frame, frame_count)
    else:
        frame_count = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret:
                frame_fun(frame, frame_count)
                frame_count += 1
            else:
                break


def convert_video_to_video(input_video_path: str, out_path: str, frame_edit_fun,
                           codec='mp4v', is_color=True):
    """
    Convert video by applying given function to each frame. Resulting video of equal height x width is
    written to the given output path.
    :param input_video_path:
    :param out_path:
    :param frame_edit_fun:
    :param codec:
    """
    # "Load" input video
    input_video = cv2.VideoCapture(input_video_path)

    # Match source video features.
    fps = input_video.get(cv2.CAP_PROP_FPS)
    size = (
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(out_path, fourcc, fps, size, is_color)

    # Process frame by frame

    # some codecs don't support this, so in such cases we need to rollback to base looping
    nb_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if nb_frames and nb_frames > 0:
        for _ in tqdm(range(nb_frames)):
            _, frame = input_video.read()

            frame = frame_edit_fun(frame)
            if frame is not None:
                out.write(frame)
    else:
        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret:
                frame = frame_edit_fun(frame)
                if frame is not None:
                    out.write(frame)
            else:
                break

    # Release everything if job is finished
    input_video.release()
    out.release()
