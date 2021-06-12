import argparse
import cv2
import imutils
import pickle
import os
import numpy as np
from algorithms.utils import createDir, getFiles
from typing import Dict, Any
from algorithms.bag_of_words.bow_descriptor import BowDescriptor
from algorithms.texture.texture_descriptor import TextureDescriptor
from algorithms.classification_based.cb_descriptor import CbDescriptor

SPRITE_SIZE = (45,45)
SPRITES_FILE = "sprites.png"
FEATURES_FILE = "features.npy"
IMAGES_FILE = "imagesPaths.npy"


def create_sprite(data):
    """
    Tile images into sprite image. 
    Add any necessary padding
    """
    
    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    
    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)
    
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3) 
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the feature vectors from a list of images")

    parser.add_argument(
        "-i", "--images_path",
        type=str,
        required=True,
        help="Path to dataset of images"
    )
    parser.add_argument(
        "-o", "--output_features_path",
        type=str,
        required=True,
        help="Path to directory to save the features"
    )
    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        required=True,
        help="Algorithmcan be either 'bow' or 'cbn' or 'lbp"
    )
    parser.add_argument(
        "-n", "--n_images",
        type=int,
        required=True,
        help="Number of images randomly sampled"
    )
    parser.add_argument(
        '--use_all', 
        default=False, 
        action='store_true'
    )
    parser.add_argument(
        '--use_color', 
        default=False, 
        action='store_true'
    )

    args: Dict[str, Any] = vars(parser.parse_args())

    createDir(args["output_features_path"])
    imagesPaths = np.array(getFiles(args["images_path"]))

    np.random.seed(2021)  
    np.random.shuffle(imagesPaths)

    descriptor = None
    print(args["algorithm"])
    if args["algorithm"] == "bow":
        descriptor = BowDescriptor()
    if args["algorithm"] == "rgb":
        descriptor = BowDescriptor()
    elif args["algorithm"] == "lbp":
        descriptor = TextureDescriptor()
    elif args["algorithm"] == "cbn":
        descriptor = CbDescriptor(
            # checkpointPath="algorithms/classification_based/models/softtriple-resnet50.pth",
            checkpointPath="algorithms/classification_based/models/resnet18_30e.pt",
            useColor=args["use_color"],
        )
        

    allFeatures = []
    imgSprites = []
    imgPaths = []

    n = len(imagesPaths)
    count = 0

    for (i, imagePath) in enumerate(imagesPaths):
        if count == args["n_images"] and not args["use_all"]:
            break
        print(i,"/",n," -> ", imagePath)
        features, image = descriptor.describe(imagePath)
        # In case len is equal to 0, means no keypoints were found for bow
        if len(features) != 0:
            if not args["use_all"]:
                sprite = cv2.resize(image, SPRITE_SIZE)
                imgSprites = imgSprites + [sprite]
            imgPaths = imgPaths + [imagePath]
            allFeatures = allFeatures + [features]
            count = count + 1
        
    
    allFeatures = np.array(allFeatures)
    imgSprites = np.array(imgSprites)
    imgPaths = np.array(imgPaths)

    if not args["use_all"]:
        allSprite = create_sprite(imgSprites)
        cv2.imwrite(os.path.join(args["output_features_path"] + '/' + SPRITES_FILE), allSprite)


    with open(os.path.join(args["output_features_path"] + '/' + FEATURES_FILE), 'wb') as f:
        np.save(f,allFeatures)
    with open(os.path.join(args["output_features_path"] + '/' + IMAGES_FILE), 'wb') as f:
        np.save(f,imgPaths)
