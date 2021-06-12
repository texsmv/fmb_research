import cv2
import pickle
import numpy as np
from .src.colorHist import getMask, getColorHist, colorHistDist
from skimage import feature
from ..utils import *
from ..color_histogram.src.color_spaces import *


def lbp_image(I: np.array):
    I= I.astype(float)

    m, n = I.shape

    z=np.zeros(8);
    b=np.zeros([m,n]);

    
    for i in range(1, m-1):
        for j in range(1, n-1):        
            t=0
            for k in range(-1,2):
                for l in range(-1,2):         
                    if (k ==0) and (l ==0):
                         continue
                    if (I[i+k][j+l]-I[i][j]<0):
                        z[t]=0
                    else:
                        z[t]=1
                    t=t+1

            for t in range(8):
                b[i][j] += ((2**t) * z[t])
    b = b.astype(np.uint8)
    return b

def ldp_image(I: np.array):
    I= I.astype(float)

    m, n = I.shape

    z=np.zeros(8);
    b=np.zeros([m,n]);

    msk=np.zeros([3,3,8])
    msk[..., 0] = np.array([[-3,-3, 5],[-3,0,5],[-3,-3,5]])
    msk[..., 1] = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    msk[..., 2] = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    msk[..., 3] = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    msk[..., 4] = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    msk[..., 5] = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    msk[..., 6] = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    msk[..., 7] = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])

    
    for i in range(1, m-1):
        for j in range(1, n-1):        
            t=0
            for k in range(-1,2):
                for l in range(-1,2):         
                    if (k == 0) and (l == 0):
                        continue
                    z[t] = I[i+k][j+l] * msk[1+k][1+l][t]
                    t=t+1
    
            q = np.argsort(z)
            g = 4

            for t in range(g):
                z[q[t]] = 0
            for t in range(g, 8):
                z[q[t]] = 1
            
            for t in range (8):
                b[i][j] = b[i][j] + (2**t * z[t])

    b = b.astype(np.uint8)
    return b


def loop_image(I: np.array):
    I= I.astype(float)

    m, n = I.shape

    x=np.zeros(8)
    y=np.zeros(8)
    b=np.zeros([m,n])

    msk=np.zeros([3,3,8])
    msk[..., 0] = np.array([[-3,-3, 5],[-3,0,5],[-3,-3,5]])
    msk[..., 1] = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    msk[..., 2] = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    msk[..., 3] = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    msk[..., 4] = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    msk[..., 5] = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    msk[..., 6] = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    msk[..., 7] = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])

    
    for i in range(1, m-1):
        for j in range(1, n-1):        
            t=0
            for k in range(-1,2):
                for l in range(-1,2):        
                    if (k == 0) and (l == 0):
                        continue
                    if I[i+k][j+l]-I[i][j]<0:
                        x[t]=0
                    else:
                        x[t]=1

                    y[t] = I[i+k][j+l] * msk[1+k][1+l][t]
                    t=t+1
    
            q = np.argsort(y)
            
            for t in range (8):
                b[i][j] = b[i][j] + ((2**(q[t])) * x[t])

    b = b.astype(np.uint8)
    return b


class TextureDescriptor:
    def __init__(self):
        self.textureHistSize = None
        self.n = 0
        self.useColor = True
        self.n_bins = 16
    
    def describe(self,imagePath: str, segmented_path: str, getImage:bool = False ):
        print(self.n)
        self.n += 1
        # print(segmented_path)
        # print(self.n)
        image = cv2.imread(imagePath)
        segmented_image :np.array = cv2.imread(segmented_path)
        target_size = 450
        width =None
        height =None
        if image.shape[0] > image.shape[1]:
            height = target_size
        else:
            width = target_size
        # print(image.shape)
        # print(segmented_image.shape)
        image = image_resize(image, width = width, height = height, inter = cv2.INTER_AREA)
        # segmented_image = image_resize(segmented_image, width = width, height = height, inter = cv2.INTER_AREA)
        segmented_image = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)
        mask :np.array = dog_mask(segmented_image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = ((gray - gray.mean()) / gray.std()) * 20 + 128 # zscore
        lbp = loop_image(gray)
        hist = cv2.calcHist([lbp], [0], mask, [256], [0, 256]).flatten()
        hist /= hist.sum()

        # print(hist)

        self.textureHistSize = len(hist)
        if self.useColor:
            # c_image = rgb_2_o1o2o3(image)
            c_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            # hist_3d = cv2.calcHist([segmented_image], [0, 1, 2], mask, [self.n_bins, self.n_bins, 3], [0, 256, 0, 256, 0, 256]).flatten()
            # hist_3d /= hist_3d.sum()
            # print(hist_3d.shape)
            # print(hist.shape)
            # return np.concatenate([hist, hist_3d])
            hist_2d = cv2.calcHist([c_image], [0, 1], mask, [self.n_bins, self.n_bins], [0, 180, 0, 256]).flatten()
            hist_2d /= hist_2d.sum()
            # print(hist_2d.shape)
            # print(hist.shape)
            return np.concatenate([hist, hist_2d])
        # hist, self.colorHistSize = self.getHistogram(image)
        if getImage:
            return hist, image
        return hist

    def compare(self, featuresA: np.ndarray, featuresB: np.ndarray):
        if self.useColor:
            text_histA = featuresA[:self.textureHistSize]
            color_histA = featuresA[self.textureHistSize:]
            text_histB = featuresB[:self.textureHistSize]
            color_histB = featuresB[self.textureHistSize:]

            text_dist = self.chi2_distance(text_histA, text_histB)
            color_dist = self.chi2_distance(color_histA, color_histB)

            # signatureA = color_histA.reshape([self.n_bins ** 3, 4]).astype(np.float32)
            # signatureB = color_histB.reshape([self.n_bins ** 3, 4]).astype(np.float32)
            # color_dist, _, _ = cv2.EMD(signatureA, signatureB, cv2.DIST_L2)
            
            return text_dist * 1 + color_dist * 1
        return self.chi2_distance(featuresA, featuresB)
        return self.descriptorDistance(featuresA, featuresB, self.colorHistSize)
    
    def getHistogram(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        textureHist = self.desc.describe(gray)
        histColor = getColorHist(image)
        return np.concatenate([textureHist, histColor]), histColor.shape[0]

    def chi2_distance(self,histA, histB, eps = 1e-10):
        # compute the chi-sqaured distance
        d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
        # return the chi-sqaured distance
        return d

    def chi2_distance_color(self, histA, histB, eps = 1e-10):
        """chi2 distance normalized, normalization base on the max posible distance

        Args:
            histA (np.ndarray): histA
            histB (np.ndarray): histN
            eps (float, optional): epsilon. Defaults to 1e-10.

        Returns:
            float: Distance in range [0, 1]
        """
        # compute the chi-sqaured distance
        countA = histA.sum()
        countB = histB.sum()
        maxDist = countA + countB

        d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

        # return the chi-sqaured distance
        return d / maxDist

    def descriptorDistance(self,histA, histB, colorHistSize):
        histCA = histA[-colorHistSize:].astype(np.float32)
        histA = histA[:-colorHistSize]
        histCB = histB[-colorHistSize:].astype(np.float32)
        histB = histB[:-colorHistSize]
        kptDist = self.chi2_distance(histA, histB)
        colorDist = self.chi2_distance_color(histCA, histCB)
        return kptDist
        # return colorDist
        return kptDist * colorDist