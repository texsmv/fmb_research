import cv2
import os
import numpy as np
import logging
from typing import Tuple, List
import matplotlib.pyplot as plt

def getFiles(dirName, allowedExtensions : Tuple = None):
    """Gets all files in the directory recursively

    Args:
        dirName (str): directory to search for files

    Returns:
        list: list of paths of files
    """
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getFiles(fullPath, allowedExtensions= allowedExtensions)
        else:
            if allowedExtensions:
                if entry.endswith(allowedExtensions):
                    allFiles.append(fullPath)
            else:
                allFiles.append(fullPath)
                
    return allFiles


def createDir(dirName:str):
    """Creates the directory if not already exists

    Args:
        dirName (str): directory path
    """
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
        logging.info(f"Created output directory {dirName}")

def getFileName(path:str):
    return path[path.rfind("/") + 1:]

def visualize_query(query_image_path: str,
                    retrieved_image_paths: List[str],
                    retrieved_distances: List[float],
                    retrieved_labels: List[str] = None,
                    query_name: str = "",
                    ncolumns = 3,
                    image_size=(224, 224)):

    n_retrieved_images: int = len(retrieved_image_paths)
    nrows: int = 2 + (n_retrieved_images - 1) // ncolumns

    _, axs = plt.subplots(nrows=nrows, ncols=ncolumns)

    # Plot query image
    query_image: np.ndarray = cv2.imread(query_image_path, cv2.IMREAD_COLOR)
    query_image = cv2.resize(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB), image_size)
    axs[0, 1].imshow(query_image)
    title: str = "Query"
    if query_name:
        title: str = f"Query: {query_name}"
    axs[0, 1].set_title(title, fontsize=14)

    # Plot retrieved images
    for i in range(n_retrieved_images):
        row: int = i // ncolumns + 1
        col: int = i % ncolumns

        retrieved_image: np.ndarray = cv2.imread(retrieved_image_paths[i], cv2.IMREAD_COLOR)
        retrieved_image = cv2.resize(cv2.cvtColor(retrieved_image, cv2.COLOR_BGR2RGB), image_size)
        distance: float = round(retrieved_distances[i], 4)
        axs[row, col].imshow(retrieved_image)

        title: str = f"N: {i + 1} Dist: {distance}" 
        if retrieved_labels:
            label: str = retrieved_labels[i]
            title = f"N: {i + 1}: {label} Dist: {distance}"
        axs[row, col].set_title(title, fontsize=10)

    # Turn off axis for all plots
    for ax in axs.ravel():
        ax.axis("off")

    return axs


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:,:,2] += value
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def dog_mask(img):
    # * color (1, 255, 0) indicates absent of dog
    mask = img != [1, 255, 0]
    mask = mask[:,:,0] * mask[:,:,1] * mask[:,:,2]
    mask = mask.astype(np.uint8) * 255
    return mask

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
