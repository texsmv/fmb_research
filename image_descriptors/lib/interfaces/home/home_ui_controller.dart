import 'dart:io';
import 'dart:ui' as ui;
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:image/image.dart' as img;
import 'package:image_descriptors/descriptors/color/o1o2o3_descriptor.dart';
import 'package:image_descriptors/descriptors/descriptors_utils.dart';
import 'package:image_descriptors/descriptors/texture/loop_descriptor.dart';
import 'package:image_descriptors/utils.dart';

class HomeUiController extends GetxController {
  O1O2O3Descriptor o1o2o3Descriptor;
  LoopDescriptor loopDescriptor;
  ByteData bytes1;
  ByteData bytes2;
  List<double> colorHistogram1;
  List<double> colorHistogram2;
  List<double> textureHistogram1;
  List<double> textureHistogram2;
  List<double> descriptor1;
  List<double> descriptor2;

  File _imageFile;
  Image imageWidget;
  ui.Image imageUi;
  img.Image _image;

  RxDouble distance = 0.0.obs;
  RxDouble colorDistance = 0.0.obs;
  RxDouble textureDistance = 0.0.obs;
  File image1;
  File image2;

  onInit() {
    o1o2o3Descriptor = O1O2O3Descriptor();
    loopDescriptor = LoopDescriptor();
  }

  Future<void> onPickImage(int position) async {
    _imageFile = await uiUtilPickImage();
    if (_imageFile == null) return;
    imageWidget = Image.file(_imageFile);
    _image = img.decodeImage(_imageFile.readAsBytesSync());
    // final img.Image resized = resizeByMaxSide(input, maxSide: 400);

    List<dynamic> mask = await dogMask(_imageFile);
    img.Image segmentedDog = applyMaskFilter(_image, mask);
    List<dynamic> colorHistogram = o1o2o3Descriptor.describe(_image, mask);
    List<dynamic> textureHistogram = loopDescriptor.describe(_image, mask);
    // print(colorHistogram);
    // print(textureHistogram);
    // List<dynamic> embedding = metric.run(_image);

    ui.Codec codec =
        await ui.instantiateImageCodec(img.encodePng(segmentedDog));
    ui.FrameInfo frameInfo = await codec.getNextFrame();
    imageUi = frameInfo.image;
    if (position == 0) {
      colorHistogram1 = colorHistogram;
      textureHistogram1 = textureHistogram;
      image1 = _imageFile;
      bytes1 = await imageUi.toByteData(format: ui.ImageByteFormat.png);
      descriptor1 = colorHistogram1 + textureHistogram1;
    } else {
      colorHistogram2 = colorHistogram;
      textureHistogram2 = textureHistogram;
      image2 = _imageFile;
      bytes2 = await imageUi.toByteData(format: ui.ImageByteFormat.png);
      descriptor2 = colorHistogram2 + textureHistogram2;
    }

    // embedding = embedding.reshape([128]);

    // if (position == 0) {
    //   embedding1 = List<double>.from(embedding);
    // } else {
    //   embedding2 = List<double>.from(embedding);
    // }

    if (descriptor1 != null && descriptor2 != null) {
      distance.value = chi2Distance(descriptor1, descriptor2);
      colorDistance.value = chi2Distance(colorHistogram1, colorHistogram2);
      textureDistance.value =
          chi2Distance(textureHistogram1, textureHistogram2);
    }
    update();
  }
}
