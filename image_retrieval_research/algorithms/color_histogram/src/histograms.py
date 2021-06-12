import numpy as np
import cv2
from .utils import dog_mask

def histogram_3d(img, mask = None, n_bins = 8):
    
    hist_3d = cv2.calcHist([img], [0, 1, 2], mask, [n_bins, n_bins, n_bins], [0, 256, 0, 256, 0, 256])
    # cv2.normalize(hist_3d, hist_3d, 0, 1, cv2.NORM_MINMAX)
    return hist_3d

def histogram_1d(img, mask = [], n_bins = 8):
    hist_channel1 = cv2.calcHist([img], [0], mask, [n_bins], [0, 256]).flatten()
    # cv2.normalize(hist_channel1, hist_channel1, 0, 1, cv2.NORM_MINMAX)
    hist_channel2 = cv2.calcHist([img], [1], mask, [n_bins], [0, 256]).flatten()
    # cv2.normalize(hist_channel2, hist_channel2, 0, 1, cv2.NORM_MINMAX)
    hist_channel3 = cv2.calcHist([img], [2], mask, [n_bins], [0, 256]).flatten()
    # cv2.normalize(hist_channel3, hist_channel3, 0, 1, cv2.NORM_MINMAX)
    hist = np.concatenate([hist_channel1, hist_channel2, hist_channel3])
    
    return hist