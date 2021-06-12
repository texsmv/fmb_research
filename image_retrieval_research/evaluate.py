from algorithms.color_histogram.hsv_descriptor import HsvDescriptor
import matplotlib.pyplot as plt
import numpy as np
import argparse
from algorithms.utils import getFiles, getFileName
from algorithms.bag_of_words.bow_descriptor import BowDescriptor
from algorithms.texture.texture_descriptor import TextureDescriptor
from algorithms.classification_based.cb_descriptor import CbDescriptor
from algorithms.color_histogram.rgb_descriptor import RgbDescriptor
from algorithms.color_histogram.o1o2o3_descriptor import O1O2O3Descriptor 
from sklearn.neighbors import BallTree
from algorithms.utils import visualize_query

import logging
logging.basicConfig(level=logging.DEBUG)


def ranking_accuracy(indices:np.array, matches_indices: list):
    N = indices.shape[0]
    accum_accuracy = 0
    for index in matches_indices:
        rank = np.where(indices == index)[0][0]
        accum_accuracy += (N - rank) / N
    accum_accuracy /= len(matches_indices)
    return accum_accuracy
    


def getFilesMatches(files:list):
    """gets the dict of images matches that follows the format "dog_{id}_{photo_id}"

    Args:
        files (list): files names

    Returns:
        (dict): dict of matches
    """
    matchs = {}
    for i, file in enumerate(files):
        if not file.startswith("dog"):
            continue
        root, dog_id, photo_id = file.split('_')
        photo_id = photo_id[:-4]
        key = f"{root}_{dog_id}"
        if key in matchs.keys():
            matchs[key] += [i]
            # matchs[key] += [f'{key}_{photo_id}.jpg']
        else:
            matchs[key] = [i]
            # matchs[key] = [f'{key}_{photo_id}.jpg']
    return matchs

def evaluate(input_path:str, segmented_path:str, algorithm:str):
    filePaths = getFiles(input_path)
    segmentedPaths = getFiles(segmented_path)
    fileNames = [getFileName(file) for file in filePaths]
    matches = getFilesMatches(fileNames)
    N = len(fileNames)

    merge = True

    if algorithm == "bow":
        descriptor = BowDescriptor()
    elif algorithm == "lbp":
        descriptor = TextureDescriptor()
    elif algorithm == "cbn":
        descriptor = CbDescriptor(
            checkpointPath="algorithms/classification_based/models/softtriple-resnet50.pth",
            # checkpointPath="algorithms/classification_based/models/resnet18_30e.pt",
            useColor=False,
            # useColor=args["use_color"],
        )
    elif algorithm == "rgb":
        descriptor = RgbDescriptor()
    elif algorithm == "hsv":
        descriptor = HsvDescriptor()
    elif algorithm == "o1o2o3":
        descriptor = O1O2O3Descriptor()
    
    if algorithm == "rgb" or algorithm == "o1o2o3" or algorithm == "hsv":
        features = [descriptor.describe(filePaths[i], segmentedPaths[i]) for i in range(len(fileNames))]
    elif algorithm == "cbn":
        features = [descriptor.describe(filePaths[i]) for i in range(len(fileNames))]
    elif algorithm == "lbp":
        features = [descriptor.describe(filePaths[i], segmentedPaths[i]) for i in range(len(fileNames))]
    else:
        features = [descriptor.describe(filePaths[i]) for i in range(len(fileNames))]
    


    features = np.array(features)
    tree = BallTree(features, metric = descriptor.compare)
    dog_ids = list(matches.keys())
    mean_accuracy = 0
    for id in dog_ids:
        index = matches[id][0]
        index_matches = matches[id][1:]
        # query_path = filePaths[query_index]
        # query_segmented_path = segmentedPaths[query_index]
        query_distances, query_indices = tree.query(np.array([features[index]]), N)
        query_indices: np.array = query_indices.ravel()
        query_distances: np.array = query_distances.ravel()
        accuracy = ranking_accuracy(query_indices, index_matches)
        mean_accuracy += accuracy



        # n_vis = 30
        # visualize_query(
        #     filePaths[index], 
        #     [filePaths[k] for k in query_indices][:n_vis], 
        #     query_distances[:n_vis], 
        #     ncolumns = 10,
        # )
        # plt.show()

    mean_accuracy /= len(dog_ids)
    print(f"Accuracy: {mean_accuracy}")





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
        "-a", "--algorithm",
        type=str,
        required=True,
        help="Algorithmcan be either 'bow' or 'cbn'"
    )

    args = vars(parser.parse_args())

    evaluate(args["input_dataset"], args["segmented_dataset"], args["algorithm"])