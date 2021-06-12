import cv2
import os
import argparse
from algorithms.utils import increase_brightness, getFiles, getFileName, createDir
import random


def changeBrightness(images, segmented_images, brightness_value = 20):
    assert len(images) == len(segmented_images)
    t_images = []
    t_segmented_images = []
    for i in range(len(images)):
        brightness_movement = random.randint(-brightness_value, brightness_value)
        image = increase_brightness(images[i], value=brightness_movement)
        segmented_image = increase_brightness(segmented_images[i], value=brightness_movement)
        t_images += [image]
        t_segmented_images += [segmented_image]
    return t_images, t_segmented_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment dogs from path")
    parser.add_argument(
        '-i', '--input_dataset',
        type=str,
        required=True,
        help='Input dataset path'
    )
    parser.add_argument(
        '-s', '--segmented_dataset',
        type=str,
        required=True,
        help='Input segmented dataset path'
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="path to save both segmented and original datasets with transforms applied"
    )

    args = vars(parser.parse_args())
    
    filePaths = getFiles(args["input_dataset"])
    fileSegmentedPaths = getFiles(args["segmented_dataset"])
    fileNames = [getFileName(file) for file in filePaths]

    createDir(args["output_path"])
    createDir(os.path.join(args["output_path"], "original"))
    createDir(os.path.join(args["output_path"], "segmented"))

    images = [cv2.imread(path) for path in filePaths]
    segmented_images = [cv2.imread(path) for path in fileSegmentedPaths]
    t_images, t_segmented_images = changeBrightness(images, segmented_images, 30)

    for i in range(len(fileNames)):
        cv2.imwrite(
            os.path.join(args["output_path"], "original", fileNames[i]), 
            t_images[i]
        )
        cv2.imwrite(
            os.path.join(args["output_path"], "segmented", fileNames[i]), 
            t_segmented_images[i]
        )

