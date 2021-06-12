import argparse
import os
import tensorflow as tf
import numpy as np
from typing import Dict, Any
from tensorflow.contrib.tensorboard.plugins import projector

SPRITES_FILE = "sprites.png"
FEATURES_FILE = "features.npy"
IMAGE_SIZE = (45,45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare for tensorflow projector")
    parser.add_argument(
        "-o", "--log_dir",
        type=str,
        required=True,
        help="Path to save logs"
    )
    args: Dict[str, Any] = vars(parser.parse_args())

    CHECKPOINT_FILE = os.path.join(args["log_dir"], "features.ckpt")
    with open(os.path.join(args["log_dir"] + "/" + FEATURES_FILE), 'rb') as f:
        features = np.load(f)
    print("-------------------------------")
    print("")
    print(features.shape)
    print("")
    print("-------------------------------")
    features = tf.Variable(features, name='features')
    with tf.Session() as sess:
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, CHECKPOINT_FILE)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        # embedding.metadata_path = METADATA_FILE

        # This adds the sprite images
        embedding.sprite.image_path = SPRITES_FILE
        embedding.sprite.single_image_dim.extend(IMAGE_SIZE)
        projector.visualize_embeddings(tf.summary.FileWriter(args["log_dir"]), config)
