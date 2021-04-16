import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
import tensorflow as tf

def edge_intensity(image, sigma):
    edges = feature.canny(image, sigma=sigma, low_threshold=None, high_threshold=None)
    ei = np.average(edges)
    return ei

def edge_intensity_tfa(image, sigma):

    tfa.image.gaussian_filter2d(image, sigma=2)
    edges = edge_intensity(image, sigma)

    ei = tf.average(edges)
    return ei

import cv2
import numpy as np
from matplotlib import pyplot as plt

def edge_intensity_cv2(image, sigma):
    edges = cv2.Canny(image,100,200)
    ei = np.average(edges)
    return ei

def average_gradient(image, tf=False):
    # Get x-gradient in "sx"
    sx = ndi.sobel(image, axis=0, mode='constant')
    # Get y-gradient in "sy"
    sy = ndi.sobel(image, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel = np.hypot(sx, sy)
    if tf:
        sx, sy = tf.image.image_gradients(tf.expand_dims(image, axis=0))

    avg_grad = np.average(np.sqrt(sx**2 + sy ** 2))

    return avg_grad