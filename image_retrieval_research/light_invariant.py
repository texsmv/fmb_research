import cv2
# import cv
import numpy as np
from matplotlib import pyplot as plt
from algorithms.color_histogram.src.color_spaces import rgb_2_c1c2c3, rgb_2_l1l2l3, rgb_2_o1o2o3
from pyemd import emd
from fastdtw import fastdtw
from algorithms.color_histogram.src.utils import *
from algorithms.color_histogram.src.histograms import *


# bright = cv2.imread("images/cube_1.png")
# dark = cv2.imread("images/cube_2.png")
bright = cv2.imread("images/boby_1.png")
dark = cv2.imread("images/boby_2.png")
# bright = cv2.imread("images/dog_14_2.jpg")
# dark = cv2.imread("images/dog_14_3.jpg")
# bright = cv2.imread("images/sub_1_2.png")
# dark = cv2.imread("images/sub_1_1.png")

# def emd (a,b):
#     earth = 0
#     earth1 = 0
#     diff = 0
#     s= len(a)
#     su = []
#     diff_array = []
#     for i in range (0,s):
#         diff = a[i]-b[i]
#         diff_array.append(diff)
#         diff = 0
#     for j in range (0,s):
#         earth = (earth + diff_array[j])
#         earth1= abs(earth)
#         su.append(earth1)
#     emd_output = sum(su)/(s-1)
#     return emd_output


def LBA_equalize(image):
    (L, A, B) = cv2.split(image)
    L = cv2.equalizeHist(L)
    return  cv2.merge([L, A, B])

def plot_histograms(image1, image2):
    hist1_c1 = cv2.calcHist([image1], [0], None, [16], [0, 256])
    hist1_c2 = cv2.calcHist([image1], [1], None, [16], [0, 256])
    hist1_c3 = cv2.calcHist([image1], [2], None, [16], [0, 256])
    
    hist2_c1 = cv2.calcHist([image2], [0], None, [16], [0, 256])
    hist2_c2 = cv2.calcHist([image2], [1], None, [16], [0, 256])
    hist2_c3 = cv2.calcHist([image2], [2], None, [16], [0, 256])


    # hist1_c3 = np.roll(hist2_c3, 3)

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].plot(hist1_c1)
    axs[0, 0].set_title("bright: L")
    axs[0, 1].plot(hist1_c2)
    axs[0, 1].set_title("bright: A")
    axs[0, 2].plot(hist1_c3)
    axs[0, 2].set_title("bright: B")
    axs[0, 3].imshow(image1)
    # axs[0, 4].imshow(cv2.cvtColor(brightLAB, cv2.COLOR_LAB2RGB) )

    axs[1, 0].plot(hist2_c1)
    axs[1, 0].set_title("dark: L")
    axs[1, 1].plot(hist2_c2)
    axs[1, 1].set_title("dark: A")
    axs[1, 2].plot(hist2_c3)
    axs[1, 2].set_title("dark: B")
    axs[1, 3].imshow(image2)

    
    # cv2.CalcEMD2(hist1_c3, hist2_c3, cv.CV_DIST_L2)
    # print("EMD")
    # print(emd(hist1_c3, hist2_c3))
    distance, path = fastdtw(hist1_c3, hist2_c3)
    print(f"dtw: {distance}")
    # axs[1, 4].imshow(cv2.cvtColor(darkLAB, cv2.COLOR_LAB2RGB) )

def plot_LAB_histograms(bright, dark, equalize:bool = True):
    brightLAB_o = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    darkLAB_o = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

    if equalize:
        brightLAB = LBA_equalize(brightLAB_o)
    else: 
        brightLAB = brightLAB_o
    hist1_c1 = cv2.calcHist([brightLAB], [0], None, [16], [0, 256])
    hist1_c2 = cv2.calcHist([brightLAB], [1], None, [16], [0, 256])
    hist1_c3 = cv2.calcHist([brightLAB], [2], None, [16], [0, 256])

    if equalize:
        darkLAB = LBA_equalize(darkLAB_o)
    else: 
        darkLAB = darkLAB_o
    hist2_c1 = cv2.calcHist([darkLAB], [0], None, [16], [0, 256])
    hist2_c2 = cv2.calcHist([darkLAB], [1], None, [16], [0, 256])
    hist2_c3 = cv2.calcHist([darkLAB], [2], None, [16], [0, 256])

    fig, axs = plt.subplots(2, 5)
    axs[0, 0].plot(hist1_c1)
    axs[0, 0].set_title("bright: L")
    axs[0, 1].plot(hist1_c2)
    axs[0, 1].set_title("bright: A")
    axs[0, 2].plot(hist1_c3)
    axs[0, 2].set_title("bright: B")
    axs[0, 3].imshow(cv2.cvtColor(brightLAB_o, cv2.COLOR_LAB2RGB) )
    axs[0, 4].imshow(cv2.cvtColor(brightLAB, cv2.COLOR_LAB2RGB) )

    axs[1, 0].plot(hist2_c1)
    axs[1, 0].set_title("dark: L")
    axs[1, 1].plot(hist2_c2)
    axs[1, 1].set_title("dark: A")
    axs[1, 2].plot(hist2_c3)
    axs[1, 2].set_title("dark: B")
    axs[1, 3].imshow(cv2.cvtColor(darkLAB_o, cv2.COLOR_LAB2RGB) )
    axs[1, 4].imshow(cv2.cvtColor(darkLAB, cv2.COLOR_LAB2RGB) )

    # plt.subplot(1, 3, 1)
    # plt.imshow('image1', bright)
    # plt.subplot(1, 3, 2)
    # plt.imshow('image2', dark)
    

# bright_c1c2c3 = rgb_2_c1c2c3(bright)
# dark_c1c2c3 = rgb_2_c1c2c3(dark)

# bright_l1l2l3 = rgb_2_l1l2l3(bright)
# dark_l1l2l3 = rgb_2_l1l2l3(dark)


# bright_l1l2l3 = rgb_2_o1o2o3(bright)
# dark_l1l2l3 = rgb_2_o1o2o3(dark)

# meanLigh1 = bright_l1l2l3[...,2].mean()
# meanLigh2 = dark_l1l2l3[...,2].mean()

# bright_l1l2l3 = bright_l1l2l3.astype(np.float32)
# dark_l1l2l3 = dark_l1l2l3.astype(np.float32)

# bright_l1l2l3 = np.dstack((
#     bright_l1l2l3[..., 0],
#     bright_l1l2l3[...,1], 
#     np.clip((bright_l1l2l3[..., 2] +  (128 - meanLigh1)) , 0, 255)
# )).astype(np.uint8)
# dark_l1l2l3 = np.dstack((
#     dark_l1l2l3[..., 0], 
#     dark_l1l2l3[...,1], 
#     np.clip((dark_l1l2l3[..., 2] +  (128 - meanLigh2)) , 0, 255)
# )).astype(np.uint8)


# plot_histograms(bright_l1l2l3, dark_l1l2l3)

# fig, axs = plt.subplots(1, 2)
# # axs[ 0].imshow(bright_c1c2c3)
# # axs[ 1].imshow(dark_c1c2c3)
# axs[ 0].imshow(bright_l1l2l3)
# axs[ 1].imshow(dark_l1l2l3)
# plt.show()


# plot_LAB_histograms(bright, dark, equalize=True)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # * ---------------------------- EMD tests ----------------------------

# def compute_histogram(src, h_bins = 30, s_bins = 32, scale = 10):
#     '''calculate histogram from picture'''
#     #create images
#     hsv = cv2.CreateImage(cv2.GetSize(src), 8, 3)
#     hplane = cv2.CreateImage(cv2.GetSize(src), 8, 1)
#     splane = cv2.CreateImage(cv2.GetSize(src), 8, 1)
#     vplane = cv2.CreateImage(cv2.GetSize(src), 8, 1)

#     planes = [hplane, splane]
#     cv2.CvtColor(src, hsv, cv2.CV_BGR2HSV)
#     cv2.CvtPixToPlane(hsv, hplane, splane, vplane, None)

#     #compute histogram
#     hist = cv2.CreateHist((h_bins, s_bins), cv2.CV_HIST_ARRAY,
#             ranges = ((0, 180),(0, 255)), uniform = True)
#     cv2.CalcHist(planes, hist)      #compute histogram
#     cv2.NormalizeHist(hist, 1.0)    #normalize histo

#     return hist



# def compute_signatures(hist1, hist2, h_bins = 30, s_bins = 32):
#     '''
#     demos how to convert 2 histograms into 2 signature
#     '''
#     num_rows = h_bins * s_bins
#     sig1 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
#     sig2 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
#     #fill signatures
#     #TODO: for production optimize this, use Numpy
#     for h in range(0, h_bins):
#         for s in range(0, s_bins):
#             bin_val = cv.QueryHistValue_2D(hist1, h, s)
#             cv.Set2D(sig1, h*s_bins + s, 0, bin_val) #bin value
#             cv.Set2D(sig1, h*s_bins + s, 1, h)  #coord1
#             cv.Set2D(sig1, h*s_bins + s, 2, s) #coord2
#             #signature.2
#             bin_val2 = cv.QueryHistValue_2D(hist2, h, s)
#             cv.Set2D(sig2, h*s_bins + s, 0, bin_val2) #bin value
#             cv.Set2D(sig2, h*s_bins + s, 1, h)  #coord1
#             cv.Set2D(sig2, h*s_bins + s, 2, s) #coord2

#     return (sig1, sig2)
# def compute_emd(src1, src2, h_bins, s_bins, scale):
#     hist1  = compute_histogram(src1, h_bins, s_bins, scale)
#     hist2  = compute_histogram(src2, h_bins, s_bins, scale)
#     sig1, sig2 = compute_signatures(hist1, hist2)
#     emd = cv.CalcEMD2(sig1, sig2, cv.CV_DIST_L2)
#     return emd

# # src = cv.LoadImage("pictures/grid1.jpg", cv.CV_LOAD_IMAGE_COLOR)
# emd = compute_emd(bright, dark, 16, 16, 10)
# # print "Current EMD: ", emd

# # * ----------------------- wasserstein_distance --------------------
# from scipy.stats import wasserstein_distance
# from imageio import imread
# import numpy as np

# def get_histogram(img):
#   '''
#   Get the histogram of an image. For an 8-bit, grayscale image, the
#   histogram will be a 256 unit vector in which the nth value indicates
#   the percent of the pixels in the image with the given darkness level.
#   The histogram's values sum to 1.
#   '''
#   print(img.shape)
#   h, w = img.shape
#   hist = [0.0] * 256
#   for i in range(h):
#     for j in range(w):
#       hist[img[i, j]] += 1
#   return np.array(hist) / (h * w)

# # bright = cv2.imread("images/sub_1_2.png")
# # dark = cv2.imread("images/sub_1_1.png")
# a = imread('images/sub_1_2.png')
# b = imread('images/sub_1_1.png')
# # a_hist = get_histogram(a)
# # b_hist = get_histogram(b)
# # print(a.shape)
# bright_o1o2o3 = rgb_2_o1o2o3(bright)
# dark_o1o2o3 = rgb_2_o1o2o3(dark)

# # image1 = moveMean(bright_o1o2o3, targetMean=150)
# # image2 = moveMean(dark_o1o2o3, targetMean=160)
# image1 = bright_o1o2o3
# image2 = dark_o1o2o3

# # a_hist = cv2.calcHist([image1], [2], None, [16], [0, 256]).flatten()
# # b_hist = cv2.calcHist([image2], [2], None, [16], [0, 256]).flatten()
# a_hist = histogram_3d(image1)
# b_hist = histogram_3d(image2)
# a_desc = histogram3d_to_signature(a_hist)
# b_desc = histogram3d_to_signature(b_hist)

# # a_desc = np.array([[i, a_hist[i]] for i in range(len(a_hist))]).astype(np.float32)
# # b_desc = np.array([[i, b_hist[i]] for i in range(len(b_hist))]).astype(np.float32)

# # print(a_hist)
# # print(b_hist)
# # emd_dist = wasserstein_distance(a_desc, b_desc)
# # emd_dist = wasserstein_distance(a_hislt, b_hist)
# emd_dist, _, _ = cv2.EMD(a_desc, b_desc, cv2.DIST_L2)
# chi2_dist = chi2_distance(a_hist.flatten(), b_hist.flatten())
# print(f"emd: {emd_dist}")
# print(f"chi2: {chi2_dist}")

# # axs[1, 3].imshow(image2)

# # fig, axs = plt.subplots(2, 2)
# # axs[0, 0].plot(a_hist)
# # axs[0, 1].imshow(image1)
# # axs[1, 0].plot(b_hist)
# # axs[1, 1].imshow(image2)

# # plt.show()


# * ------------------ Histograms-------------------------------

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


brigth_gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
dark_gray = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)

brigth_lbp = lbp_image(brigth_gray)
dark_lbp = lbp_image(dark_gray)
brigth_hist = cv2.calcHist([brigth_lbp], [0], None, [256], [0, 256]).flatten()
dark_hist = cv2.calcHist([dark_lbp], [0], None, [256], [0, 256]).flatten()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(brigth_hist)
axs[0, 1].imshow(brigth_lbp)
axs[1, 0].plot(dark_hist)
axs[1, 1].imshow(dark_lbp)
plt.show()

