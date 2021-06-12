import argparse
import os
import numpy as np
import faiss
import cv2
import matplotlib.pyplot as plt
from algorithms.bag_of_words.bow_descriptor import BowDescriptor
from algorithms.texture.texture_descriptor import TextureDescriptor
from algorithms.classification_based.cb_descriptor import CbDescriptor
from typing import Dict, Any, List
from sklearn.neighbors import BallTree
from algorithms.utils import getFiles

SPRITES_FILE = "sprites.png"
FEATURES_FILE = "features.npy"
IMAGES_FILE = "imagesPaths.npy"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare for tensorflow projector")
    parser.add_argument(
        "-o", "--log_dir",
        type=str,
        required=True,
        help="Path to save logs"
    )
    parser.add_argument(
        "-i", "--images_path",
        type=str,
        required=True,
        help="Path of the query image"
    )

    parser.add_argument(
        "-a", "--algorithm",
        type=str,
        required=True,
        help="Algorithmcan be either 'bow' or 'cbn'"
    )

    parser.add_argument(
        "-k", "--k_neighbors",
        type=int,
        required=True,
        help="Number of k nearest neighbors"
    )

    parser.add_argument(
        '--use_color', 
        default=False, 
        action='store_true'
    )

    args: Dict[str, Any] = vars(parser.parse_args())

    imagesPaths = getFiles(args["images_path"])

    with open(os.path.join(args["log_dir"] + "/" + FEATURES_FILE), 'rb') as f:
        features = np.load(f)
        features = features.astype(np.float32)
    with open(os.path.join(args["log_dir"] + "/" + IMAGES_FILE), 'rb') as f:
        imgPaths = np.load(f)

    descriptor = None
    if args["algorithm"] == "bow":
        descriptor = BowDescriptor()
    elif args["algorithm"] == "lbp":
        descriptor = TextureDescriptor()
    elif args["algorithm"] == "cbn":
        descriptor = CbDescriptor(
            checkpointPath="algorithms/classification_based/models/softtriple-resnet50.pth",
            # checkpointPath="algorithms/classification_based/models/resnet18_30e.pt",
            useColor=args["use_color"],
        )
    
    tree = BallTree(features, metric = descriptor.compare)
    for imagePath in imagesPaths:
        embedding, _ = descriptor.describe(imagePath)
        distances, indices = tree.query(np.array([embedding]), args["k_neighbors"])
        indices: List[int] = indices.ravel().tolist()
        distances: List[float] = distances.ravel().tolist()
        query_paths: List[str] = [imgPaths[i] for i in indices]
        visualize_query(
            imagePath, 
            query_paths, 
            distances, 
            ncolumns = 10,
        )
        plt.show()