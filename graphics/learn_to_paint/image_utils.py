import cv2


def get_phase_and_magnitude(img, sobel_kernel_size=7):
    """
    Calculate phase/rotation angle from image gradient
    :param img: image to compute phase from
    :param sobel_kernel_size:
    :return: phase in float32 radian
    """
    # grayify
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

    # gradient (along x and y axis)
    xg = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    yg = - cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)

    # calculates the rotation angle of the 2D vectors gradients
    phase = cv2.phase(xg, yg)

    # calculates the magnitude of the 2D vectors gradients
    magnitude = cv2.magnitude(xg, yg)
    magnitude = magnitude / magnitude.max()  # normalize to [0, 1] range

    return phase, magnitude
