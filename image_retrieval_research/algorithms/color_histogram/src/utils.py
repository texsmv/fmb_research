import numpy as np

def dog_mask(img):
    # * color (1, 255, 0) indicates absent of dog
    mask = img != [1, 255, 0]
    mask = mask[:,:,0] * mask[:,:,1] * mask[:,:,2]
    mask = mask.astype(np.uint8) * 255
    return mask

def chi2_distance_normalized(histA, histB, eps = 1e-10):
    countA = histA.sum()
    countB = histB.sum()
    maxDist = countA + countB

    # compute the chi-sqaured distance
    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    # return the chi-sqaured distance
    return d / maxDist

def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-sqaured distance
    countA = histA.sum()
    countB = histB.sum()
    maxDist = countA + countB

    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    # return the chi-sqaured distance
    return d

def zScoreChannel(image, channel = 2, targetMean = 128):
    channel = image[...,channel]
    img = image.astype(np.float32)
    channel = ((channel - channel.mean()) / channel.std()) * 20 + 128 # zscore
    
    img = np.dstack((
        img[..., 0],
        img[...,1], 
        channel,
    )).astype(np.uint8)
    return img

def moveMean(image, channel = 2, targetMean = 128):
    mean = image[...,channel].mean()
    img = image.astype(np.float32)
    
    img = np.dstack((
        img[..., 0],
        img[...,1], 
        np.clip((img[..., 2] +  (targetMean - mean)) , 0, 255)
    )).astype(np.uint8)
    return img


def histogram3d_to_signature(hist):
    bin_1, bin_2, bin_3 = hist.shape 
    num_rows = bin_1 * bin_2 * bin_3
    sig = np.zeros((num_rows, 4), np.float32)
    for i in range(bin_1):
        for j in range(bin_2):
            for k in range(bin_3):
                sig[i * (bin_2 * bin_3) + j * (bin_3) + k] = np.array([hist[i][j][k], i, j, k])
    return sig

def histogram2d_to_signature(hist):
    bin_1, bin_2 = hist.shape 
    num_rows = bin_1 * bin_2
    sig = np.zeros((num_rows, 3), np.float32)
    for i in range(bin_1):
        for j in range(bin_2):
            sig[i * (bin_1) + j] = np.array([hist[i][j], i, j])
    return sig
