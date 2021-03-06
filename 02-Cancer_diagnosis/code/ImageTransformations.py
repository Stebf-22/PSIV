import cv2
from scipy import ndimage
import random 
import numpy as np


def basic_operation(img: np.ndarray, operation: str, kernel_shape: tuple = (5, 5), iterations: int = 1) -> np.ndarray:
    """
    Basic morphological operations

    :param img: input image
    :param operation: erode/dilate/opening/closing/morphGradient operation
    :param kernel_shape: kernel shape of the operation (n, m)
    :param iterations: number of iterations to perform the operation
    """

    assert img.ndim == 2, "Not a 2-dimensional image"
    kernel = np.ones(kernel_shape, dtype=np.uint8)

    if operation == "erode":
        return cv2.erode(img.copy(), kernel, iterations)

    elif operation == "dilate":
        return cv2.dilate(img.copy(), kernel, iterations)

    elif operation == "opening":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape[0])
        return cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel, iterations)

    elif operation == "closing":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape[0])
        return cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel)

    elif operation == "morphGradient":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        return cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel)

    else:
        raise NotImplementedError("Operation not implemented")


def resize(img, des_size: tuple = (30, 60)):
    return cv2.resize(img, des_size)


def rotate(img: np.ndarray, angle: float):
    return ndimage.rotate(img, angle)


def gaussian_noise(img: np.ndarray, mean: float = 0.2, stddev: float = 0.3):
    noise = stddev * np.random.randn(*img.shape) + mean
    return img + noise


def scale(img, fx, fy):
    return resize(cv2.resize(img, img.shape, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC))


def prime_otsu(image, is_normalized = True):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

def mnorm(img, mean, std):
    return (img - mean)/std

def otsu(img):
    thresh = prime_otsu(img)
    return img > thresh



def down_sample_up_sample(img):
    img = resize(img, (20, 20))
    return resize(img)


def morph_operation(img: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations randomly to the input image. 50% for each individual operation:
    rotation, dilation or erosion and gaussian noise filter.

    :param img: Input image
    :return: Rotated, dilated/eroded & gaussian noise input image
    """

    img = resize(img)

    # Rotation & resizing (lambda < 0.5)
    rotation_probability = random.random()
    max_rotation, min_rotation = 15, 0
    rotation_angle = random.random() * (max_rotation - min_rotation) + min_rotation
    if rotation_probability <= 0.15:
        img = rotate(img, int(rotation_angle))
        down_up_prob = random.random()
        if down_up_prob < 0.5:
            img = down_sample_up_sample(img)

    # Dilation / Erosion (lambda < 0.5)
    max_kernel , min_kernel = 2, 1
    kernel_size_X = int(random.random() * (max_kernel-min_kernel)) + min_kernel
    kernel_size_Y = int(random.random() * (max_kernel-min_kernel)) + min_kernel
    kernel = (kernel_size_X, kernel_size_Y)

    dilate_erode_probability = random.random()

    if dilate_erode_probability < 0.5:
        operation = "dilate" if dilate_erode_probability < 0.25 else "erode"
        img = basic_operation(img, operation, kernel)

    # Gaussian Noise filter (lambda > 0.5)
    max_mean, min_mean = 0.05, 0
    max_stddev, min_stddev = 0.1, 0.01
    mean = random.random() * (max_mean - min_mean) + min_mean
    stddev = random.random() * (max_stddev - min_stddev) + min_stddev

    if random.random() > 0.5:
        img = gaussian_noise(img, mean=mean, stddev=stddev)

    _, final_image = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return final_image
