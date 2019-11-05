import cv2


def load_imgs_from_videos(dirpath, img_size, frame_idxs_to_keep=None):
    videos_paths = list(map(str, dirpath.glob("*.mp4")))

    imgs = []
    for video in videos_paths:
        input_video = cv2.VideoCapture(video)
        idx = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret:
                if frame_idxs_to_keep is None or idx in frame_idxs_to_keep:
                    imgs.append(cv2.resize(frame, img_size))
                idx += 1
            else:
                break

    return imgs
