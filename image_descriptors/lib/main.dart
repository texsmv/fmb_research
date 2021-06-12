import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_navigation/src/root/get_material_app.dart';
import 'package:image_descriptors/interfaces/home/home_ui.dart';
import 'package:image_descriptors/interfaces/home/home_ui_controller.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomeUi(),
      onInit: () {
        Get.put(HomeUiController(), permanent: true);
      },
    );
  }
}
