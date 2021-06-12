import numpy as np
import tensorflow as tf
import os
import sys
import cv2
from PIL import Image
from imutils import paths
from six.moves import urllib

from algorithms.bag_of_words.src.deepLabModel import DeepLabModel


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True

# from bagOfWords.deepLabModel import DeepLabModel

from algorithms.utils import getFiles

INPUT_DATASET = "data/archive/images/Images"
OUTPUT_DATASET = "algorithms/bagOfWords/segmentedData"

MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = "algorithms/bagOfWords/models"
tf.io.gfile.makedirs(model_dir)
download_path = os.path.join(model_dir, _TARBALL_NAME)

if __name__ == '__main__':
    MODEL = []
    try:
        MODEL = DeepLabModel(download_path)
    except (OSError, IOError) as e:
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                        download_path)
        print('download completed! loading DeepLab model...')
        MODEL = DeepLabModel(download_path)


    if not os.path.isdir(OUTPUT_DATASET):
        os.mkdir(OUTPUT_DATASET)

    images = getFiles(INPUT_DATASET)

    for i, image in enumerate(images):
        if not os.path.exists(os.path.join(OUTPUT_DATASET, image[image.rfind("/") + 1:])):
            print(i, "/", len(images))

            pil_image = Image.open(image)
            
            img, mask = MODEL.getMaskFromClass(pil_image, 12)

            cv_img = img.convert('RGB')
            open_cv_image = np.array(cv_img)
            open_cv_image = open_cv_image[:, :, ::-1].copy() 


            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channel mask

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if np.array_equal(mask[i][j], np.array([0, 0, 0], dtype = np.uint8)):
                        open_cv_image[i][j] = np.array([0, 255, 0], dtype = np.uint8)

            
            # open_cv_image = cv2.bitwise_and(open_cv_image,mask)

            cv2.imwrite(os.path.join(OUTPUT_DATASET, image[image.rfind("/") + 1:]) , open_cv_image)