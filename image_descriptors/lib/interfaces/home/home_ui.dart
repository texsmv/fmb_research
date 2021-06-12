import 'dart:typed_data';

import 'package:charts_painter/chart.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_descriptors/interfaces/home/home_ui_controller.dart';

class HomeUi extends StatefulWidget {
  HomeUi({Key key}) : super(key: key);

  @override
  _HomeUiState createState() => _HomeUiState();
}

class _HomeUiState extends State<HomeUi> {
  HomeUiController controller = Get.find();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: SizedBox.expand(
        child: GetBuilder<HomeUiController>(
          builder: (_) => Column(
            children: [
              Expanded(
                child: Column(
                  children: [
                    Expanded(
                      child: Row(
                        children: [
                          Expanded(
                            child: controller.image1 != null
                                ? Image.memory(
                                    Uint8List.view(controller.bytes1.buffer))
                                : SizedBox(),
                          ),
                          Expanded(
                            child: controller.colorHistogram1 != null
                                ? Column(
                                    children: [
                                      Expanded(
                                        child: Chart(
                                          state: ChartState.bar(
                                            ChartData.fromList(
                                              controller.colorHistogram1
                                                  .map((e) => BarValue<void>(e))
                                                  .toList(),
                                            ),
                                          ),
                                        ),
                                      ),
                                      Expanded(
                                        child: Chart(
                                          state: ChartState.bar(
                                            ChartData.fromList(
                                              controller.textureHistogram1
                                                  .map((e) => BarValue<void>(e))
                                                  .toList(),
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  )
                                : SizedBox(),
                          )
                        ],
                      ),
                    ),
                    SizedBox(
                      child: TextButton(
                        onPressed: () {
                          controller.onPickImage(0);
                        },
                        child: Text("Pick image"),
                      ),
                    ),
                  ],
                ),
              ),
              Obx(() => Column(
                    children: [
                      Text("Color distance: ${controller.colorDistance.obs}"),
                      Text(
                          "Texture distance: ${controller.textureDistance.obs}"),
                      Text("Total distance: ${controller.distance.obs}"),
                    ],
                  )),
              Expanded(
                child: Column(
                  children: [
                    Expanded(
                      child: Row(
                        children: [
                          Expanded(
                            child: controller.image2 != null
                                ? Image.memory(
                                    Uint8List.view(controller.bytes2.buffer))
                                : SizedBox(),
                          ),
                          Expanded(
                            child: controller.colorHistogram2 != null
                                ? Column(
                                    children: [
                                      Expanded(
                                        child: Chart(
                                          state: ChartState.bar(
                                            ChartData.fromList(
                                              controller.colorHistogram2
                                                  .map((e) => BarValue<void>(e))
                                                  .toList(),
                                            ),
                                          ),
                                        ),
                                      ),
                                      Expanded(
                                        child: Chart(
                                          state: ChartState.bar(
                                            ChartData.fromList(
                                              controller.textureHistogram2
                                                  .map((e) => BarValue<void>(e))
                                                  .toList(),
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  )
                                : SizedBox(),
                          )
                        ],
                      ),
                    ),
                    SizedBox(
                      child: TextButton(
                        onPressed: () {
                          controller.onPickImage(1);
                        },
                        child: Text("Pick image"),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
