import 'dart:io';

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';

class PickImageBottomSheet extends StatefulWidget {
  final bool multiple;
  const PickImageBottomSheet({Key key, this.multiple = false})
      : super(key: key);

  @override
  _PickImageBottomSheetState createState() => _PickImageBottomSheetState();
}

class _PickImageBottomSheetState extends State<PickImageBottomSheet> {
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 135,
      child: Column(
        children: [
          // const BottomSheetDragger(),
          GestureDetector(
            onTap: () {
              _pickImage(ImageSource.gallery);
            },
            behavior: HitTestBehavior.opaque,
            child: SizedBox(
              height: 50,
              child: Row(
                children: const [
                  // Padding(
                  //   padding: EdgeInsets.symmetric(horizontal: 16),
                  //   child: ImageIcon(
                  //     pIconGalery,
                  //     size: 17,
                  //     color: pPrimaryTextColor,
                  //   ),
                  // ),
                  Text(
                    "Galería",
                    style: TextStyle(
                      // color: pPrimaryTextColor,
                      fontSize: 14,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                ],
              ),
            ),
          ),
          GestureDetector(
            onTap: () {
              _pickImage(ImageSource.camera);
            },
            behavior: HitTestBehavior.opaque,
            child: SizedBox(
              height: 50,
              child: Row(
                children: const [
                  // Padding(
                  //   padding: EdgeInsets.symmetric(horizontal: 16),
                  //   child: ImageIcon(
                  //     pIconCamera,
                  //     size: 17,
                  //     color: pPrimaryTextColor,
                  //   ),
                  // ),
                  Text(
                    "Cámara",
                    style: TextStyle(
                      // color: pPrimaryTextColor,
                      fontSize: 14,
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    final ImagePicker _imagePicker = ImagePicker();
    // final PickedFile pickedfile = await _imagePicker.getImage(
    //   source: source,
    //   maxHeight: 400,
    //   maxWidth: 400,
    // );
    List<XFile> images;
    if (widget.multiple) {
      images = [
        await _imagePicker.pickImage(
          source: source,
          maxWidth: 400,
          maxHeight: 400,
        )
      ];
      if (images[0] == null) return Get.back(result: null);
    } else {
      images = await _imagePicker.pickMultiImage(
        // source: source,
        maxWidth: 400,
        maxHeight: 400,
      );
      if (images == null) return Get.back(result: null);
    }

    // if (pickedfile == null) return Get.back(result: null);
    return Get.back(
      result: List.generate(
        images.length,
        (index) => File(images[index].path),
      ),
    );
  }
}
