import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_state.dart';
import 'package:stats/stats.dart';

import '../home_ui_controller.dart';

class ChartsView extends GetView<HomeUiController> {
  ChartsView({Key key}) : super(key: key);

  List<Stats> _stats;

  @override
  Widget build(BuildContext context) {
    return GetBuilder<HomeUiController>(builder: (_) {
      _stats = controller.getStats();
      if (_stats.isEmpty) return Container();
      List<double> _heights = _averageHeights();
      print(_heights);
      return Center(
        child: Container(
          height: 300,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Text(
                    "color",
                    style: TextStyle(
                      color: Colors.red,
                      fontSize: 18,
                    ),
                  ),
                  Expanded(child: SizedBox()),
                  Container(
                    width: 20,
                    height: _heights[0],
                    color: Colors.red,
                  ),
                  Text(
                    "average: ${_stats[0].average}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "std: ${_stats[0].standardDeviation}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "min: ${_stats[0].min}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "max: ${_stats[0].max}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
              Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Text(
                    "texture",
                    style: TextStyle(
                      color: Colors.blue,
                      fontSize: 18,
                    ),
                  ),
                  Expanded(child: SizedBox()),
                  Container(
                    width: 20,
                    height: _heights[1],
                    color: Colors.blue,
                  ),
                  Text(
                    "average: ${_stats[1].average}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "std: ${_stats[1].standardDeviation}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "min: ${_stats[1].min}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                  Text(
                    "max: ${_stats[1].max}",
                    style: TextStyle(
                      fontSize: 15,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      );
    });
  }

  int _maxStatPosition() {
    if (_stats[0].average > _stats[1].average) {
      return 0;
    }

    if (_stats[1].average > _stats[0].average) {
      return 1;
    }
  }

  List<double> _averageHeights() {
    int maxPosition = _maxStatPosition();
    double _maxHeight = 200;
    return List.generate(
        _stats.length,
        (index) =>
            _stats[index].average * _maxHeight / _stats[maxPosition].average);
  }
}
