import os
import tensorflow as tf
import argparse
import torch
from torch.autograd import Variable
import numpy as np
from typing import Dict, Any
from algorithms.classification_based.src.resnet import Resnet50, Resnet18
import torch.onnx
import onnx
from torchvision import models
from onnx_tf.backend import prepare
from algorithms.utils import createDir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pytorch model to tensorflow lite")
    parser.add_argument(
        "-m", "--checkpoint",
        type=str,
        required=True,
        help="Pytorch model"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output path"
    )
    parser.add_argument(
        "-n", "--model_name",
        type=str,
        required=True,
        help="Model name"
    )
    args: Dict[str, Any] = vars(parser.parse_args())
    checkpoint: Dict[str, Any] = torch.load(args["checkpoint"], map_location="cpu")
    model = Resnet50(128)
    model.load_state_dict(checkpoint["model_state_dict"])
    input_np = np.random.uniform(0, 1, (1, 3, 448, 448))
    input_var = Variable(torch.FloatTensor(input_np))
    model_name = args['model_name']
    output_dir = args['output_dir']

    createDir(output_dir)
    
    torch.onnx.export(
        model,
        input_var,
        os.path.join(output_dir, f'{model_name}.onnx'),
        input_names = ['input'],
        output_names = ['output'],
    )

    onnx_model = onnx.load(os.path.join(output_dir, f'{model_name}.onnx'))

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(os.path.join(output_dir, f'{model_name}.pb'))

    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(output_dir, f'{model_name}.pb'))
    tflite_model = converter.convert()

    with open(os.path.join(output_dir, f'{model_name}.tflite'), 'wb') as f:
        f.write(tflite_model)


    # k_model = onnx_to_keras(onnx_model, ['input'])
    # print("exported")



    # test = model(input_var)
    # print(test.shape)

    # k_model = pytorch_to_keras(model, input_var)
    