import 'dart:io';
import 'package:get/get.dart';

import 'package:image_cropper/image_cropper.dart';
import 'package:flutter/material.dart';
import 'package:image_descriptors/pick_image_bottom_sheet.dart';

Future<File> uiUtilcropImage(
  File imageFile,
  String cropperTitle, {
  List<CropAspectRatioPreset> aspectRatios = const [
    CropAspectRatioPreset.ratio4x3,
  ],
  Color color = Colors.blue,
}) async {
  ImageCropper cropper = ImageCropper();
  final File croppedFile = await cropper.cropImage(
    sourcePath: imageFile.path,
    aspectRatioPresets: aspectRatios,
    androidUiSettings: AndroidUiSettings(
      toolbarTitle: cropperTitle,
      toolbarWidgetColor: Colors.white,
      initAspectRatio: aspectRatios[0],
      lockAspectRatio: true,
      toolbarColor: color,
      activeControlsWidgetColor: color,
    ),
    iosUiSettings: IOSUiSettings(
      title: cropperTitle,
      aspectRatioLockEnabled: true,
    ),
  );
  if (croppedFile != null) {
    return croppedFile;
  }
  return null;
}

Future<List<File>> uiUtilPickImages({
  bool crop = false,
  String cropperTitle = "Cropper",
  bool verifyDog = false,
  bool multiple = false,
  List<CropAspectRatioPreset> aspectRatios = const [
    CropAspectRatioPreset.ratio4x3,
  ],
  Color color = Colors.blue,
}) async {
  List<File> images = await showModalBottomSheet(
    context: Get.context,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(15)),
    ),
    builder: (context) => PickImageBottomSheet(
      multiple: multiple,
    ),
  );

  if (crop && !multiple && images != null) {
    images[0] = await uiUtilcropImage(images[0], cropperTitle,
        aspectRatios: aspectRatios, color: color);
  }
  return images;
}

Future<void> uiUtilDelayed(VoidCallback callback,
    {Duration delay = const Duration(milliseconds: 250)}) async {
  await Future.delayed(delay);
  callback();
}

void uiUtilShowLoaderOverlay() {
  Get.dialog(WillPopScope(
      onWillPop: () async => false,
      child: const Center(child: CircularProgressIndicator())));
}

void uiUtilHideLoaderOverlay() {
  Get.back();
}
