import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../home_ui_controller.dart';
import 'dog_card.dart';

class FiguresView extends GetView<HomeUiController> {
  const FiguresView({Key key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 20, vertical: 4),
      child: Column(
        children: [
          SizedBox(height: 20),
          Column(
            children: [
              AspectRatio(
                aspectRatio: 7 / 4,
                child: GetBuilder<HomeUiController>(
                  id: "query",
                  builder: (_) => DogCard(model: controller.queryModel),
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  SizedBox(
                    child: TextButton(
                      onPressed: () {
                        controller.onPickQueryDog();
                      },
                      child: Text("Query image"),
                    ),
                  ),
                  SizedBox(
                    child: TextButton(
                      onPressed: () {
                        controller.onPickDatabaseDog();
                      },
                      child: Text("Dataset image"),
                    ),
                  ),
                ],
              ),
            ],
          ),
          Divider(),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Column(
                children: [
                  Text(
                    "Color weight:",
                    style: TextStyle(
                      color: Colors.red,
                    ),
                  ),
                  SizedBox(height: 5),
                  Container(
                    color: Colors.white,
                    width: 60,
                    height: 35,
                    child: TextField(
                      controller: controller.colorController,
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(
                        contentPadding:
                            EdgeInsets.symmetric(vertical: 14, horizontal: 3),
                      ),
                    ),
                  ),
                ],
              ),
              Column(
                children: [
                  Text(
                    "Texture weight:",
                    style: TextStyle(
                      color: Colors.blue,
                    ),
                  ),
                  SizedBox(height: 5),
                  Container(
                    color: Colors.white,
                    width: 60,
                    height: 35,
                    child: TextField(
                      controller: controller.textureController,
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(
                        contentPadding:
                            EdgeInsets.symmetric(vertical: 14, horizontal: 3),
                      ),
                    ),
                  ),
                ],
              ),
              Column(
                children: [
                  Text(
                    "Threshold:",
                    style: TextStyle(
                      color: Colors.black,
                    ),
                  ),
                  SizedBox(height: 5),
                  Container(
                    color: Colors.white,
                    width: 60,
                    height: 35,
                    child: TextField(
                      controller: controller.thresholdController,
                      keyboardType: TextInputType.number,
                      decoration: InputDecoration(
                        contentPadding:
                            EdgeInsets.symmetric(vertical: 14, horizontal: 3),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          Expanded(
            child: Column(
              children: [
                Expanded(
                  child: GetBuilder<HomeUiController>(
                    id: "dataset",
                    builder: (_) => ListView.builder(
                      padding: EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 10,
                      ),
                      itemBuilder: (context, index) {
                        return Row(
                          children: [
                            Expanded(
                              child: AspectRatio(
                                aspectRatio: 7 / 4,
                                child: DogCard(
                                  model: controller.datasetModels[index],
                                ),
                              ),
                            ),
                            Icon(
                              controller.doesDogPassThreshold(
                                      controller.datasetModels[index])
                                  ? Icons.check
                                  : Icons.cancel,
                              color: controller.doesDogPassThreshold(
                                      controller.datasetModels[index])
                                  ? Colors.green
                                  : Colors.red,
                            )
                          ],
                        );
                      },
                      itemCount: controller.datasetModels.length,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
