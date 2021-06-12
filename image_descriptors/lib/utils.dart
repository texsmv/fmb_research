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
  final File croppedFile = await ImageCropper.cropImage(
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

Future<File> uiUtilPickImage({
  bool crop = false,
  String cropperTitle = "Cropper",
  bool verifyDog = false,
  List<CropAspectRatioPreset> aspectRatios = const [
    CropAspectRatioPreset.ratio4x3,
  ],
  Color color = Colors.blue,
}) async {
  File image = await showModalBottomSheet(
    context: Get.context,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(15)),
    ),
    builder: (context) => const PickImageBottomSheet(),
  );

  if (crop && image != null) {
    image = await uiUtilcropImage(image, cropperTitle,
        aspectRatios: aspectRatios, color: color);
  }
  // if (verifyDog && image != null) {
  //   if (!await DogDetector.hasADog(image)) {
  //     uiUtilDelayed(() {
  //       Get.snackbar("Smart center", "Boby no detectado.");
  //     });
  //     return null;
  //   }
  // }

  return image;
}

Future<void> uiUtilDelayed(VoidCallback callback,
    {Duration delay = const Duration(milliseconds: 250)}) async {
  await Future.delayed(delay);
  callback();
}
