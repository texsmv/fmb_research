import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_overlay_loader/flutter_overlay_loader.dart';
import 'package:get/get.dart';
import 'package:image/image.dart' as img;
import 'package:image_descriptors/descriptors/deep_metric/deep_metric.dart';
import 'package:image_descriptors/descriptors/descriptors_utils.dart';
import 'package:image_descriptors/models/dog_model.dart';
import 'package:image_descriptors/utils.dart';
import 'package:loader_overlay/loader_overlay.dart';
import 'package:oktoast/oktoast.dart';
import 'package:stats/stats.dart';

class HomeUiController extends GetxController {
  // * States
  DogModel get queryModel => _queryModel;
  List<DogModel> get datasetModels => _datasetsModels;
  List<double> get distancesWeights => _distancesWeights;

  // * Controllers
  PageController get pageController => _pageController;
  TextEditingController get textureController => _textureController;
  TextEditingController get colorController => _colorController;
  TextEditingController get thresholdController => _thresholdController;

  @override
  onInit() {
    super.onInit();
    deepMetric = DeepMetric();
    _datasetsModels = [];
    _pageController = PageController();
  }

  Future<void> onPickQueryDog() async {
    List<DogModel> models = await _onPickImage();
    if (models != null) {
      _queryModel = models[0];
      _computeAllDatasetDistances();
      _sortDogModels();
      update(["query"]);
      update(["dataset"]);
      update();
    }
  }

  Future<void> onPickDatabaseDog() async {
    final List<DogModel> models = await _onPickImage();
    if (models == null) return;

    _datasetsModels.addAll(models);
    if (_queryModel != null) {
      for (var i = 0; i < models.length; i++) {
        final double distance = _dogDistances(_queryModel, models[i]);

        models[i].deepDistance = distance;
      }
    }
    _sortDogModels();
    update(["dataset"]);
    update();
  }

  void _computeAllDatasetDistances() {
    if (_queryModel == null) return;
    if (_datasetsModels.isEmpty) return;

    for (var i = 0; i < _datasetsModels.length; i++) {
      final double distance = _dogDistances(_queryModel, _datasetsModels[i]);
      _datasetsModels[i].deepDistance = distance;
    }
  }

  /// Sort the dogsModels base on the distance
  void _sortDogModels() {
    if (queryModel == null) return;
    datasetModels.sort((dogA, dogB) {
      return dogA.distance.compareTo(dogB.distance);
    });
  }

  Future<List<DogModel>> _onPickImage() async {
    Get.context.loaderOverlay.show();
    final List<File> imageFiles = await uiUtilPickImages();
    if (imageFiles == null) {
      Get.context.loaderOverlay.hide();
      return null;
    }

    // final Image imageWidget = Image.file(imageFile);
    // final img.Image resized = resizeByMaxSide(input, maxSide: 400);
    List<DogModel> models = [];
    for (var i = 0; i < imageFiles.length; i++) {
      showToast("Loaded ${i + 1} dogs");
      DogModel dogModel = await _createDogModel(imageFiles[i]);
      models.add(dogModel);
    }
    Get.context.loaderOverlay.hide();
    return models;
  }

  Future<DogModel> _createDogModel(
    File imageFile,
  ) async {
    img.Image image = img.decodeImage(imageFile.readAsBytesSync());

    // *Used to calculate the deep metric
    final List<dynamic> deepDescriptor = deepMetric.run(image);

    final ui.Codec codec = await ui.instantiateImageCodec(img.encodePng(image));

    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    final ui.Image imageUi = frameInfo.image;
    final ByteData bytes =
        await imageUi.toByteData(format: ui.ImageByteFormat.png);

    final DogModel model = DogModel(
      // colorHistogram: colorHistogram,
      // textureHistogram: textureHistogram,
      bytes: bytes,
      deepDescriptor: deepDescriptor,
      // distancesWeights: _distancesWeights,
    );
    return model;
  }

  /// returns the color, texture and deepMetric distance in that order
  double _dogDistances(DogModel dogA, DogModel dogB) {
    final double deepDistance =
        euclideanDistance(dogA.deepDescriptor, dogB.deepDescriptor);
    print("Deep distance: $deepDistance");

    return deepDistance;
  }

  // * Descriptors
  DeepMetric deepMetric;

  DogModel _queryModel;
  List<DogModel> _datasetsModels;
  List<double> _distancesWeights;

  double _threshold;

  // * Controllers
  PageController _pageController;
  TextEditingController _textureController;
  TextEditingController _colorController;
  TextEditingController _thresholdController;
}
