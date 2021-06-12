import os
import numpy as np
import sys
import cv2
import tensorflow as tf
from PIL import Image
import argparse
from segmentation.deeplab_model import DeepLabModel
from algorithms.utils import getFiles
from six.moves import urllib



MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
TARBALL_NAME = 'deeplab_model.tar.gz'

def segment_dataset(input_path:str, output_path:str):
    model_dir = "algorithms/models"
    tf.io.gfile.makedirs(model_dir)
    download_path = os.path.join(model_dir, TARBALL_NAME)

    MODEL = []
    try:
        MODEL = DeepLabModel(download_path)
    except (OSError, IOError) as e:
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(DOWNLOAD_URL_PREFIX + MODEL_URLS[MODEL_NAME],
                        download_path)
        print('download completed! loading DeepLab model...')
        MODEL = DeepLabModel(download_path)


    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    image_paths = getFiles(input_path)
    for i, path in enumerate(image_paths):
        file = path[path.rfind("/") + 1:]
        if not os.path.exists(os.path.join(output_path, file)):
            print(i, "/", len(image_paths))
            image = Image.open(os.path.join(input_path, file))
            img, mask = MODEL.getMaskFromClass(image, 12)
            cv_img = np.array(img.convert('RGB'))
            cv_img = cv_img[:, :, ::-1].copy()

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if np.array_equal(mask[i][j], np.array([0, 0, 0], dtype = np.uint8)):
                        cv_img[i][j] = np.array([0, 255, 0], dtype = np.uint8)
            
            cv2.imwrite(os.path.join(output_path, file) , cv_img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment dogs from path")
    parser.add_argument(
        '-i', '--input_dataset',
        type=str,
        required=True,
        help='Input dataset path'
    )
    parser.add_argument(
        '-o', '--output_dataset',
        type=str,
        required=True,
        help='Output path for segmented photos'
    )

    args = vars(parser.parse_args())
    segment_dataset(args['input_dataset'], args['output_dataset'])
