import cv2
import pickle
import numpy as np
from .src.bagOfWords import BagOfVisualWords
from .src.detectAndDescribe import DetectAndDescribe
from .src.colorHist import getMask, getColorHist, colorHistDist
from ..utils import *


class BowDescriptor:
    def __init__(self, colorHistSize: int = 512):
        self.centers = pickle.load(open("algorithms/bag_of_words/src/clusterCenters.pickle","rb"))
        self.detector = DetectAndDescribe()
        self.BoW = BagOfVisualWords(self.centers)
        self.colorHistSize = colorHistSize
    
    def describe(self,imagePath: str, getImage:bool = False):
        image = cv2.imread(imagePath)
        hist, self.colorHistSize = self.getHistogram(image)
        if getImage:
            return hist, image
        return hist

    def compare(self, featuresA: np.ndarray, featuresB: np.ndarray):
        return self.descriptorDistance(featuresA, featuresB, self.colorHistSize)
    
    def getHistogram(self, image):
        width =None
        height =None
        target_size = 450
        if image.shape[0] > image.shape[1]:
            height = target_size
        else:
            width = target_size

        image = image_resize(image, width = width, height = height, inter = cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kpt, descr = self.detector.describe(gray)
        if kpt.any() ==None or descr.any() == None:
            print("Error, no keypoints detected")
            return [], None

        histKpts = self.BoW.describe(descr)
        histKpts /= histKpts.sum()
        histColor = getColorHist(image)
        histColor /= histColor.sum()
        print(histKpts.shape)
        print(histColor.shape)
        return np.concatenate([histKpts, histColor]), histColor.shape[0]

    def chi2_distance(self,histA, histB, eps = 1e-10):
        # compute the chi-sqaured distance
        d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
        # return the chi-sqaured distance
        return d

        

    def descriptorDistance(self,histA, histB, colorHistSize):
        histCA = histA[-colorHistSize:].astype(np.float32)
        histA = histA[:-colorHistSize]
        histCB = histB[-colorHistSize:].astype(np.float32)
        histB = histB[:-colorHistSize]
        kptDist = self.chi2_distance(histA, histB)
        colorDist = colorHistDist(histCA, histCB)
        # return kptDist
        # return colorDist`
        return kptDist * colorDist