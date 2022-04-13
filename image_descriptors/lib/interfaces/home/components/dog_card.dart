import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_descriptors/models/dog_model.dart';

class DogCard extends StatefulWidget {
  final DogModel model;
  const DogCard({Key key, @required this.model}) : super(key: key);

  @override
  _DogCardState createState() => _DogCardState();
}

class _DogCardState extends State<DogCard> {
  DogModel get model => widget.model;

  @override
  Widget build(BuildContext context) {
    if (model == null)
      return Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(15),
          color: Colors.white,
        ),
      );
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      child: Container(
        padding: EdgeInsets.all(15),
        child: Column(
          children: [
            (model.deepDistance != null)
                ? Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        "Deep distance: ${model.distance.toStringAsFixed(3)}",
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  )
                : SizedBox(),
            Expanded(
              child: Row(
                children: [
                  Expanded(
                    child: Container(
                      height: double.infinity,
                      child: Image.memory(
                        Uint8List.view(model.bytes.buffer),
                        fit: BoxFit.fill,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
