import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

double euclideanDistance(List<double> vector1, List<double> vector2) {
  assert(vector1.length == vector2.length);
  double sum = 0;
  for (var i = 0; i < vector1.length; i++) {
    sum += pow(vector1[i] - vector2[i], 2);
  }
  return sqrt(sum);
}

double chi2Distance(List<double> vector1, List<double> vector2) {
  assert(vector1.length == vector2.length);
  double sum = 0;
  for (var i = 0; i < vector1.length; i++) {
    sum +=
        pow(vector1[i] - vector2[i], 2) / (vector1[i] + vector2[i] + 0.00001);
  }

  return 0.5 * sum;
}

img.Image resizeByOneSide(img.Image image, {int width, int height}) {
  // initialize the dimensions of the image to be resized and
  // grab the image size
  int targetWidth;
  int targetHeight;
  int h = image.height;
  int w = image.width;

  // if both the width and height are null, then return the
  // original image
  if (width == null && height == null) {
    return image;
  }

  // check to see if the width is null
  if (width == null) {
    // calculate the ratio of the height and construct the
    // dimensions
    double r = height / (h).toDouble();
    targetWidth = (w * r).toInt();
    targetHeight = height;
  }

  // otherwise, the height is null
  else {
    // calculate the ratio of the width and construct the
    // dimensions
    double r = width / (w).toDouble();
    targetWidth = width;
    targetHeight = (h * r).toInt();
  }
  final img.Image resized =
      img.copyResize(image, width: targetWidth, height: targetHeight);

  return resized;
}

img.Image resizeByMaxSide(img.Image image, {int maxSide = 450}) {
  int targetWidth, targetHeight;
  if (image.width > image.height) {
    targetWidth = maxSide;
  } else {
    targetHeight = maxSide;
  }
  img.Image resized =
      resizeByOneSide(image, width: targetWidth, height: targetHeight);
  return resized;
}

img.Image applyMaskFilter(img.Image original, List<dynamic> mask) {
  img.Image thumbnail =
      img.copyResize(original, width: original.width, height: original.height);
  int r, g, b;
  for (var i = 0; i < thumbnail.width; i++) {
    for (var j = 0; j < thumbnail.height; j++) {
      r = original.getPixelSafe(i, j) >> 0 & 0x000000FF;
      g = original.getPixelSafe(i, j) >> 8 & 0x000000FF;
      b = original.getPixelSafe(i, j) >> 16 & 0x000000FF;
      if (mask[i][j] == true) {
        thumbnail.setPixel(i, j, original.getPixel(i, j));
      } else {
        thumbnail.setPixelRgba(i, j, r - 120, g - 120, b - 60, 20);
      }
    }
  }
  return thumbnail;
}

extension ColorSpaces on img.Image {
  List<dynamic> toList() {
    final List<dynamic> rgbList =
        List.filled(height * width * 3, 0).reshape([width, height, 3]);
    int r, g, b;
    for (var i = 0; i < width; i++) {
      for (var j = 0; j < height; j++) {
        if (getPixelSafe(i, j) != 4278255360) {
          r = getPixelSafe(i, j) >> 0 & 0x000000FF;
          g = getPixelSafe(i, j) >> 8 & 0x000000FF;
          b = getPixelSafe(i, j) >> 16 & 0x000000FF;
        }
        rgbList[i][j] = [r, g, b];
      }
    }
    return rgbList;
  }
}

extension ListShape on List {
  /// Reshape list to a another [shape]
  ///
  /// [T] is the type of elements in list
  ///
  /// Returns List<dynamic> if [shape.length] > 5
  /// else returns list with exact type
  ///
  /// Throws [ArgumentError] if number of elements for [shape]
  /// mismatch with current number of elements in list
  List reshape<T>(List<int> shape) {
    var dims = shape.length;
    var numElements = 1;
    for (var i = 0; i < dims; i++) {
      numElements *= shape[i];
    }

    if (numElements != computeNumElements) {
      throw ArgumentError(
          'Total elements mismatch expected: $numElements elements for shape: $shape but found $computeNumElements');
    }

    if (dims <= 5) {
      switch (dims) {
        case 2:
          return this._reshape2<T>(shape);
        case 3:
          return this._reshape3<T>(shape);
        case 4:
          return this._reshape4<T>(shape);
        case 5:
          return this._reshape5<T>(shape);
      }
    }

    var reshapedList = flatten<dynamic>();
    for (var i = dims - 1; i > 0; i--) {
      var temp = [];
      for (var start = 0;
          start + shape[i] <= reshapedList.length;
          start += shape[i]) {
        temp.add(reshapedList.sublist(start, start + shape[i]));
      }
      reshapedList = temp;
    }
    return reshapedList;
  }

  List<List<T>> _reshape2<T>(List<int> shape) {
    var flatList = flatten<T>();
    List<List<T>> reshapedList = List.generate(
      shape[0],
      (i) => List.generate(
        shape[1],
        (j) => flatList[i * shape[1] + j],
      ),
    );

    return reshapedList;
  }

  List<List<List<T>>> _reshape3<T>(List<int> shape) {
    var flatList = flatten<T>();
    List<List<List<T>>> reshapedList = List.generate(
      shape[0],
      (i) => List.generate(
        shape[1],
        (j) => List.generate(
          shape[2],
          (k) => flatList[i * shape[1] * shape[2] + j * shape[2] + k],
        ),
      ),
    );

    return reshapedList;
  }

  List<List<List<List<T>>>> _reshape4<T>(List<int> shape) {
    var flatList = this.flatten<T>();

    List<List<List<List<T>>>> reshapedList = List.generate(
      shape[0],
      (i) => List.generate(
        shape[1],
        (j) => List.generate(
          shape[2],
          (k) => List.generate(
            shape[3],
            (l) => flatList[i * shape[1] * shape[2] * shape[3] +
                j * shape[2] * shape[3] +
                k * shape[3] +
                l],
          ),
        ),
      ),
    );

    return reshapedList;
  }

  List<List<List<List<List<T>>>>> _reshape5<T>(List<int> shape) {
    var flatList = flatten<T>();
    List<List<List<List<List<T>>>>> reshapedList = List.generate(
      shape[0],
      (i) => List.generate(
        shape[1],
        (j) => List.generate(
          shape[2],
          (k) => List.generate(
            shape[3],
            (l) => List.generate(
              shape[4],
              (m) => flatList[i * shape[1] * shape[2] * shape[3] * shape[4] +
                  j * shape[2] * shape[3] * shape[4] +
                  k * shape[3] * shape[4] +
                  l * shape[4] +
                  m],
            ),
          ),
        ),
      ),
    );

    return reshapedList;
  }

  /// Get shape of the list
  List<int> get shape {
    if (isEmpty) {
      return [];
    }
    var list = this as dynamic;
    var shape = <int>[];
    while (list is List) {
      shape.add((list as List).length);
      list = list.elementAt(0);
    }
    return shape;
  }

  /// Flatten this list, [T] is element type
  /// if not specified List<dynamic> is returned
  List<T> flatten<T>() {
    var flat = <T>[];
    forEach((e) {
      if (e is List) {
        flat.addAll(e.flatten());
      } else if (e is T) {
        flat.add(e);
      } else {
        // Error with typing
      }
    });
    return flat;
  }

  /// Get the total number of elements in list
  int get computeNumElements {
    var n = 1;
    for (var i = 0; i < shape.length; i++) {
      n *= shape[i];
    }
    return n;
  }
}
