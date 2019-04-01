import numpy as np
import cv2
from tqdm import tqdm

# Option of relying on MoviePy (http://zulko.github.io/moviepy/index.html)


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
        for _ in tqdm(range(nb_frames)):
            _, frame = input_video.read()
            frame_fun(frame)
    else:
        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret:
                frame_fun(frame)
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
