// import 'package:flutter/material.dart';
import 'dart:math';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart';

class DeepMetric {
  final String _modelFile = 'resnet50.tflite';
  Interpreter _interpreter;
  final int imageSize = 448;
  final int embeddingSize = 128;
  TensorImage _inputImage;
  List<int> _inputShape;
  List<int> _outputShape;
  TfLiteType _outputType;
  TensorBuffer _outputBuffer;
  Image procesedImage;

  DeepMetric() {
    _loadModel();
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset(_modelFile);
    print('Deep metric loaded successfully');
    _interpreter.allocateTensors();
    _inputShape = _interpreter.getInputTensor(0).shape;
    _outputShape = _interpreter.getOutputTensor(0).shape;
    _outputType = _interpreter.getOutputTensor(0).type;
    _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);

    print("Input shape: ${_interpreter.getInputTensors()}");
    print("Output shape: ${_interpreter.getOutputTensors()}");
    print("Output type: $_outputType");
  }

  // List<dynamic> run(List<dynamic> input) {
  //   var output = List<double>(embeddingSize).reshape([1, embeddingSize]);
  //   _interpreter.run(input, output);
  //   return output;
  // }

  TensorImage _preProcess() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    print(_inputShape[2]);
    print(_inputShape[3]);
    return ImageProcessorBuilder()
        // .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(imageSize, imageSize, ResizeMethod.NEAREST_NEIGHBOUR))
        .add(
          NormalizeOp.multipleChannels(
            [0.406, 0.456, 0.485],
            [0.225, 0.224, 0.229],
            // [0.485, 0.456, 0.406],
            // [0.229, 0.224, 0.225],
          ),
        )
        .build()
        .process(_inputImage);
  }

  List<dynamic> run(Image input) {
    _inputImage = TensorImage.fromImage(input);
    _inputImage = _preProcess();
    // procesedImage = _inputImage.getTensorBuffer().getIntList();
    var output = List<double>(embeddingSize).reshape([1, embeddingSize]);
    // print(_inputImage.getHeight());
    // print(_inputImage.getWidth());
    print(_inputImage.getTensorBuffer().getDoubleList().shape);
    print("Buffer shape");
    print(_inputImage.getTensorBuffer().getShape());
    print("--------------------");
    List<dynamic> inputList = _inputImage.getTensorBuffer().getDoubleList();
    List<dynamic> inputListT = transpose(inputList);
    // _interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer());
    _interpreter.run(inputListT, _outputBuffer.getBuffer());
    print(_outputBuffer.getDoubleList());
    return _outputBuffer.getDoubleList();
  }

  List<dynamic> transpose(List<dynamic> input) {
    print(input.shape);
    input = input.reshape([448, 448, 3]);
    List<dynamic> output = List.generate(448 * 448 * 3, (index) => 0.0);
    output = output.reshape([3, 448, 448]);

    for (var i = 0; i < 448; i++) {
      for (var j = 0; j < 448; j++) {
        for (var k = 0; k < 3; k++) {
          output[k][i][j] = input[i][j][k];
          // output[k][i][j] = 0.485;
          // output[k][i][j] = 0.0;
        }
      }
    }
    return output.reshape([1, 3, 448, 448]);
    // print(input.shape);
  }

  List<dynamic> get tensorTest =>
      List.generate(3 * imageSize * imageSize, (index) => 0.485)
          .reshape([1, 3, imageSize, imageSize]);
}
