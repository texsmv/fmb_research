import cv2
import pickle
import numpy as np
from features.bagOfWords import BagOfVisualWords
from features.detectAndDescribe import DetectAndDescribe
from features.colorHist import getMask, getColorHist, colorHistDist

DATASET = "SmallDataset"

centers = pickle.load(open("clusterCenters.pickle","rb"))



def getHistogram(image, detector, BoW):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kpt, descr = detector.describe(gray)
    if kpt.any() ==None or descr.any() == None:
        print("Error, no keypoints detected")
        return np.array([None])

    histKpts = BoW.describe(descr)
    histColor = getColorHist(image)
    return np.concatenate([histKpts, histColor]), histColor.shape[0]


def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-sqaured distance
    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    # return the chi-sqaured distance
    return d

    
def descriptorDistance(histA, histB, colorHistSize):
    histCA = histA[-colorHistSize:].astype(np.float32)
    histA = histA[:-colorHistSize]
    histCB = histB[-colorHistSize:].astype(np.float32)
    histB = histB[:-colorHistSize]
    kptDist = chi2_distance(histA, histB)
    colorDist = colorHistDist(histCA, histCB)
    print("Kpts Dist: ", kptDist, "ColorDist: ", colorDist)
    return kptDist * colorDist 




# if __name__ == "__init__":
detector = DetectAndDescribe()
BoW = BagOfVisualWords(centers)
image = cv2.imread(DATASET + "/n02085620_199.jpg")

hist, colorHistSize = getHistogram(image, detector, BoW)

print(descriptorDistance(hist, hist, colorHistSize))

