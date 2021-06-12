import cv2
from .src.histograms import *
from .src.utils import *

class HsvDescriptor:
    def __init__(self):
        self.hist_size = 0
        self.n_bins = 32
    
    def describe(self, path:str, segmented_path:str):
        image :np.array = cv2.imread(path)
        segmented_image :np.array = cv2.imread(path)
        mask :np.array = dog_mask(segmented_image)

        segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2HSV)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # hist_3d :np.array = histogram_3d(image, mask = mask, n_bins = self.n_bins)
        hist_2d = cv2.calcHist([image], [0, 1], None, [self.n_bins, self.n_bins], [0, 180, 0, 256])

        # hist :np.array = histogram_1d(image, mask = mask, n_bins = self.n_bins)
        # # signature = histogram2d_to_signature(hist_2d)
        # signature = histogram3d_to_signature(hist_3d)
        # # signature = signature.reshape([self.n_bins ** 2 * 3])
        # signature = signature.reshape([self.n_bins ** 3 * 4])

        # return signature
        # return hist
        return hist_2d.flatten()
        # return hist_3d.flatten()
    
    def compare(self, featuresA: np.array, featuresB: np.array):
        # signatureA = featuresA.reshape([self.n_bins ** 2, 3]).astype(np.float32)
        # signatureB = featuresB.reshape([self.n_bins ** 2, 3]).astype(np.float32)
        # signatureA = featuresA.reshape([self.n_bins ** 3, 4]).astype(np.float32)
        # signatureB = featuresB.reshape([self.n_bins ** 3, 4]).astype(np.float32)
        # emd, _, _ = cv2.EMD(signatureA, signatureB, cv2.DIST_L2)
        # return emd
        return chi2_distance_normalized(featuresA, featuresB)
