import numpy as np
import cv2

def rgb_2_c1c2c3(img):
    im = img.astype(np.float32)+0.001
    c1c2c3 = np.arctan(
        im/np.dstack(
            (
                cv2.max(im[...,1], im[...,2]), 
                cv2.max(im[...,0], im[...,2]), 
                cv2.max(im[...,0], im[...,1])
            )
        )
    )
    return c1c2c3

# def rgb_2_l1l2l3(img):
#     img = img.astype(np.float32)+0.001
#     R_G = np.power(img[..., 0] - img[..., 1], 2)
#     G_B = np.power(img[..., 1] - img[..., 2], 2)
#     B_R = np.power(img[..., 2] - img[..., 0], 2)

#     l1 = R_G / (R_G + G_B + B_R )
#     l2 = B_R / (R_G + G_B + B_R )
#     l3 = G_B / (R_G + G_B + B_R )
#     # print(l1.max())
#     # print(l2.max())
#     # print(l3.max())
#     # print(l1.min())
#     # print(l2.min())
#     # print(l3.min())
#     return np.dstack((l1, l2, l3))
#     # c1c2c3 = np.arctan(
#     #     im/np.dstack(
#     #         (
#     #             cv2.max(im[...,1], im[...,2]), 
#     #             cv2.max(im[...,0], im[...,2]), 
#     #             cv2.max(im[...,0], im[...,1])
#     #         )
#     #     )
#     # )
#     # return c1c2c3

def rgb_2_l1l2l3(img):
    img = img.astype(np.float32)+0.001
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # l1 = (R + G + B) / 3
    # l2 = (R - B) / 2
    # l3 = (2 * G - R - B) / 4
    l1 = (R + G + B) / 3.0
    l2 = (R - B + 255) / 2.0
    l3 = (2.0 * G - R - B + 510) / 4.0
    # print(R[0] - B[0])
    # print(l1.max())
    # print(l1.min())
    # print(l2.min())
    # print(l2.max())
    # print(l3.max())
    # print(l3.min())
    return np.dstack((l1, l2, l3)).astype(np.uint8)

def rgb_2_o1o2o3(img):
    img = img.astype(np.float32)+0.001
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    o1 = (255 + G - R) / 2.0
    o2 = (510 + R + G - (2 * B)) / 4.0
    o3 = (R + G + B) / 3.0

    # print(o1.max())
    # print(o1.min())
    # print(o2.min())
    # print(o2.max())
    # print(o3.max())
    # print(o3.min())
    return np.dstack((o1, o2, o3)).astype(np.uint8)