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
            children: [],
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
                        return AspectRatio(
                          aspectRatio: 4 / 3,
                          child: DogCard(
                            model: controller.datasetModels[index],
                          ),
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
