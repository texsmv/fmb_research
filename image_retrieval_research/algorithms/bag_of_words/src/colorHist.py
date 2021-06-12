import numpy as np
import cv2

# DATASET = "SmallDatasetSeg"

# img = cv2.imread(DATASET + "/n02102318_3415.jpg")
# img2 = cv2.imread(DATASET + "/n02106550_3886.jpg")
# img3 = cv2.imread(DATASET + "/n02106550_3931.jpg")


def chi2_distance_color(histA, histB, eps = 1e-10):
    # compute the chi-sqaured distance
    countA = histA.sum()
    countB = histB.sum()
    maxDist = countA + countB

    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    # return the chi-sqaured distance
    return d / maxDist

def getMask(img):
    # todo: check if [1, 255, 0] is equal for all masked images    
    mask = img != [1, 255, 0]
    mask = mask[:,:,0] * mask[:,:,1] * mask[:,:,2]
    mask = mask.astype(np.uint8) * 255
    return mask

def getColorHist(img):
    mask = getMask(img)

    # hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], mask, [20, 30], [0, 180, 0, 256])

    hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256] )
    # hist = cv2.normalize(hist, hist).flatten()
    hist = hist.flatten()

    
    return hist

def colorHistDist(hist1, hist2):
    # return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return chi2_distance_color(hist1, hist2)



# if __name__ == "__init__":

# hist1 = getColorHist(img)
# hist2 = getColorHist(img2)
# hist3 = getColorHist(img3)

# print(hist1.dtype)
# print(hist2.dtype)

# # d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# d = colorHistDist(hist1, hist2)
# print(d) 