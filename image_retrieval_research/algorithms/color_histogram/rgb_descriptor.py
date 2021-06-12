import cv2
import random
from .src.histograms import *
from .src.utils import *
from ..utils import *

class RgbDescriptor:
    def __init__(self):
        self.hist_size = 0
        self.n_bins = 4
        self.n = 0
    
    def describe(self, path:str, segmented_path:str):
        print(self.n)
        self.n += 1
        image :np.array = cv2.imread(path)
        segmented_image :np.array = cv2.imread(path)

        width =None
        height =None
        target_size = 450
        if image.shape[0] > image.shape[1]:
            height = target_size
        else:
            width = target_size

        image = image_resize(image, width = width, height = height, inter = cv2.INTER_AREA)
        segmented_image = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)

        mask :np.array = dog_mask(segmented_image)

        # hist_3d :np.array = histogram_3d(segmented_image, mask = mask, n_bins = self.n_bins)
        # hist_3d = hist_3d.flatten()
        # hist_3d /= hist_3d.sum()
        hist :np.array = histogram_1d(image, mask = mask, n_bins = self.n_bins)
        hist = hist.flatten()
        hist /= hist.sum()

        # signature = histogram3d_to_signature(hist_3d)
        # signature = signature.reshape([self.n_bins ** 3 * 4])
        # print(signature.shape)

        # return signature
        return hist
        print(hist_3d.shape)
        return hist_3d
    
    def compare(self, featuresA: np.array, featuresB: np.array):
        # signatureA = featuresA.reshape([self.n_bins ** 3, 4]).astype(np.float32)
        # signatureB = featuresB.reshape([self.n_bins ** 3, 4]).astype(np.float32)
        # emd, _, _ = cv2.EMD(signatureA, signatureB, cv2.DIST_L2)
        # return emd
        return chi2_distance(featuresA, featuresB)
