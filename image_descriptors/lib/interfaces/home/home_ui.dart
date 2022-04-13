import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_descriptors/interfaces/home/components/figures_view.dart';
import 'package:image_descriptors/interfaces/home/home_ui_controller.dart';
import 'package:loader_overlay/loader_overlay.dart';

class HomeUi extends StatefulWidget {
  HomeUi({Key key}) : super(key: key);

  @override
  _HomeUiState createState() => _HomeUiState();
}

class _HomeUiState extends State<HomeUi> {
  HomeUiController controller = Get.find();

  @override
  Widget build(BuildContext context) {
    return LoaderOverlay(
      child: Scaffold(
        backgroundColor: Color.fromRGBO(245, 245, 245, 1),
        body: SizedBox.expand(
          child: FiguresView(),
        ),
      ),
    );
  }
}
