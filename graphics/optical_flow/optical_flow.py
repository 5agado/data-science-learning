import argparse
import sys
import numpy as np
import cv2

from ds_utils.video_utils import generate_video

# see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def optical_flow_kanade(input_video_path):
    # load input video
    cap = cv2.VideoCapture(input_video_path)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            frames.append(img)
            frame_count += 1
        else:
            break

    cap.release()
    return frames

def dense_optical_flow(input_video_path):
    cap = cv2.VideoCapture(input_video_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            prvs = next
            frames.append(frame)
            frame_count += 1
        else:
            break

    cap.release()
    return frames


def show_video(frames):
    for frame in frames:
        cv2.imshow('frame2', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

def main(_=None):
    parser = argparse.ArgumentParser(description='Optical Flow')
    parser.add_argument('-i', '--input', required=True, help="input path")
    parser.add_argument('-o', '--out', help="output path")
    #parser.add_argument('--width', default=-1)
    #parser.add_argument('--height', default=-1)
    parser.add_argument('--fps', default=24)
    parser.add_argument('--codec', default='mp4v')
    parser.add_argument('--method', default='dense')

    args = parser.parse_args()
    method = args.method
    out_path = args.out

    if method == 'dense':
        frames = dense_optical_flow(args.input)
    elif method == 'kanade':
        frames = optical_flow_kanade(args.input)
    else:
        print(f'No such method: {method}')
        sys.exit(0)

    if out_path:
        generate_video(out_path, frames[0].shape[:2][::-1], lambda i: frames[i],
                       len(frames), fps=int(args.fps), is_color=True, codec=args.codec)
    else:
        show_video(frames)

if __name__ == "__main__":
    main(sys.argv[1:])
