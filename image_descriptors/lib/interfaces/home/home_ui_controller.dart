import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_overlay_loader/flutter_overlay_loader.dart';
import 'package:get/get.dart';
import 'package:image/image.dart' as img;
import 'package:image_descriptors/descriptors/color/o1o2o3_descriptor.dart';
import 'package:image_descriptors/descriptors/deep_metric/deep_metric.dart';
import 'package:image_descriptors/descriptors/descriptors_utils.dart';
import 'package:image_descriptors/descriptors/texture/loop_descriptor.dart';
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
    _datasetsModels = [];

    o1o2o3Descriptor = O1O2O3Descriptor();
    loopDescriptor = LoopDescriptor();

    _distancesWeights = [1.0, 5.0];

    _pageController = PageController();

    _threshold = 0.55;
    _thresholdController = TextEditingController(text: _threshold.toString());
    _thresholdController.addListener(() {
      _threshold = double.tryParse(_thresholdController.text) ?? _threshold;
      update(["dataset"]);
    });

    _colorController =
        TextEditingController(text: _distancesWeights[0].toString());
    _textureController =
        TextEditingController(text: _distancesWeights[1].toString());
    _colorController.addListener(() {
      _distancesWeights[0] =
          double.tryParse(_colorController.text) ?? _distancesWeights[0];
      _computeAllDatasetDistances();
      _sortDogModels();
      update(["dataset"]);
      update();
    });
    _textureController.addListener(() {
      _distancesWeights[1] =
          double.tryParse(_textureController.text) ?? _distancesWeights[1];
      _computeAllDatasetDistances();
      _sortDogModels();
      update(["dataset"]);
      update();
    });
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
        final List<double> distances = _dogDistances(_queryModel, models[i]);
        models[i].colorDistance = distances[0];
        models[i].textureDistance = distances[1];
      }
    }
    _sortDogModels();
    update(["dataset"]);
    update();
  }

  bool doesDogPassThreshold(DogModel dog) {
    return dog.distance < _threshold;
  }

  List<Stats> getStats() {
    if (_datasetsModels.isEmpty || _queryModel == null) return [];
    final List<double> colorDistances = List.generate(_datasetsModels.length,
        (index) => _datasetsModels[index].colorDistance * _distancesWeights[0]);
    final List<double> textureDistances = List.generate(
        _datasetsModels.length,
        (index) =>
            _datasetsModels[index].textureDistance * _distancesWeights[1]);

    return [
      Stats.fromData(colorDistances).withPrecision(3),
      Stats.fromData(textureDistances).withPrecision(3),
    ];
  }

  void _computeAllDatasetDistances() {
    if (_queryModel == null) return;
    if (_datasetsModels.isEmpty) return;

    for (var i = 0; i < _datasetsModels.length; i++) {
      final List<double> distances =
          _dogDistances(_queryModel, _datasetsModels[i]);
      _datasetsModels[i].colorDistance = distances[0];
      _datasetsModels[i].textureDistance = distances[1];
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

    // Segment dog
    List<dynamic> mask = await dogMask(imageFile);
    img.Image segmentedDog = applyMaskFilter(image, mask);

    // Get descriptors
    final List<dynamic> colorHistogram = o1o2o3Descriptor.describe(image, mask);
    final List<dynamic> textureHistogram = loopDescriptor.describe(image, mask);
    // *Used to calculate the deep metric
    // final List<dynamic> deepDescriptor = deepMetric.run(image);

    final ui.Codec codec =
        await ui.instantiateImageCodec(img.encodePng(segmentedDog));
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    final ui.Image imageUi = frameInfo.image;
    final ByteData bytes =
        await imageUi.toByteData(format: ui.ImageByteFormat.png);

    final DogModel model = DogModel(
      colorHistogram: colorHistogram,
      textureHistogram: textureHistogram,
      bytes: bytes,
      distancesWeights: _distancesWeights,
    );
    return model;
  }

  /// returns the color, texture and deepMetric distance in that order
  List<double> _dogDistances(DogModel dogA, DogModel dogB) {
    final double colorDistance =
        chi2Distance(dogA.colorHistogram, dogB.colorHistogram);
    final double textureDistance =
        chi2Distance(dogA.textureHistogram, dogB.textureHistogram);

    return [colorDistance, textureDistance];
  }

  // * Descriptors
  O1O2O3Descriptor o1o2o3Descriptor;
  LoopDescriptor loopDescriptor;
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
