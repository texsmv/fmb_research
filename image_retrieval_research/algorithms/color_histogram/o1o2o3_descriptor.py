import cv2
from .src.histograms import *
from .src.utils import *
from .src.color_spaces import rgb_2_l1l2l3, rgb_2_o1o2o3
from ..utils import *

class O1O2O3Descriptor:
    def __init__(self):
        self.hist_size = 0
        self.bins_3d = [8, 8, 8]
        self.bins_2d = [16, 16]
        self.bins_1d = 16
        self.mode = "2d"
        self.distance = "chi2"
        # self.distance = "emd"
        self.color_space = "l1l2l3"
        # self.color_space = "hsv"
        self.firstChannelRangeEnd = 256
    
    def describe(self, path:str, segmented_path:str):
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
        mask:np.array = dog_mask(segmented_image)

        if self.color_space == "o1o2o3":
            segmented_image = rgb_2_o1o2o3(segmented_image)
            image = rgb_2_o1o2o3(image)

        elif self.color_space == "l1l2l3":
            segmented_image = rgb_2_l1l2l3(segmented_image)
            image = rgb_2_l1l2l3(image)

        elif self.color_space == "hsv":
            segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2HSV)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            self.firstChannelRangeEnd = 180


        # image = moveMean(image)
        # image = zScoreChannel(image)
        # segmented_image = moveMean(segmented_image)
        feature = None
        if self.mode == "3d":
            hist_3d = cv2.calcHist([image], [0, 1, 2], mask, self.bins_3d, [0, self.firstChannelRangeEnd, 0, 256, 0, 256])
            hist_3d /= hist_3d.sum()
            if self.distance == "chi2":
                hist_3d = hist_3d.flatten()
                feature = hist_3d
            elif self.distance == "emd":
                signature = histogram3d_to_signature(hist_3d).flatten()
                feature = signature
        elif self.mode == "2d":
            hist_2d = cv2.calcHist([image], [0, 1], mask, self.bins_2d, [0, self.firstChannelRangeEnd, 0, 256])
            hist_2d /= hist_2d.sum()
            if self.distance == "chi2":
                hist_2d = hist_2d.flatten()
                feature = hist_2d
            elif self.distance == "emd":
                signature = histogram2d_to_signature(hist_2d).flatten()
                feature = signature
        elif self.mode == "1d":
            hist_1d :np.array = histogram_1d(image, mask = mask, n_bins = self.bins_1d)
            hist_1d /= hist_1d.sum()
            self.distance = "chi2"
            hist_1d = hist_1d.flatten()
            feature = hist_1d
            
        print(feature.shape)
        return feature

        # hist :np.array = histogram_1d(image, mask = mask, n_bins = self.n_bins)

        
    
    def compare(self, featuresA: np.array, featuresB: np.array):
        if self.mode == "1d":
            distance =  chi2_distance(featuresA, featuresB)
        elif self.distance == "chi2":
            distance =  chi2_distance(featuresA, featuresB)
        elif self.distance == "emd":
            if self.mode == "2d":
                signatureA = featuresA.reshape([self.bins_2d[0] * self.bins_2d[1], 3]).astype(np.float32)
                signatureB = featuresB.reshape([self.bins_2d[0] * self.bins_2d[1], 3]).astype(np.float32)
            elif self.mode == "3d":
                signatureA = featuresA.reshape([self.bins_3d[0] * self.bins_3d[1] * self.bins_3d[2], 4]).astype(np.float32)
                signatureB = featuresB.reshape([self.bins_3d[0] * self.bins_3d[1] * self.bins_3d[2], 4]).astype(np.float32)
            emd, _, _ = cv2.EMD(signatureA, signatureB, cv2.DIST_L2)
            distance = emd
        return distance
