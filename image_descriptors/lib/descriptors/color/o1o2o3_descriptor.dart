import 'package:image/image.dart' as img;
import 'package:image_descriptors/descriptors/descriptors_utils.dart';

class O1O2O3Descriptor {
  final List<int> bins = [16, 16];

  List<dynamic> rgbTo1o2o3(List<dynamic> rgbImage) {
    final int width = rgbImage.shape[0];
    final int height = rgbImage.shape[1];
    final List<dynamic> o1o2o3Image =
        List.filled(height * width * 3, 0).reshape([width, height, 3]);
    int r, g, b;
    for (var i = 0; i < width; i++) {
      for (var j = 0; j < height; j++) {
        r = rgbImage[i][j][0];
        g = rgbImage[i][j][1];
        b = rgbImage[i][j][2];
        o1o2o3Image[i][j][0] = (255.0 + g - r) ~/ 2;
        o1o2o3Image[i][j][1] = ((510.0 + r + g - (2 * b)) ~/ 4.0);
        o1o2o3Image[i][j][2] = (r + g + b) ~/ 3.0;
      }
    }
    return o1o2o3Image;
  }

  List<double> describe(img.Image input, List<dynamic> mask) {
    // final img.Image resized = resizeByMaxSide(input, maxSide: 400);
    final List<dynamic> rgbList = input.toList();
    final List<dynamic> o1o2o3List = rgbTo1o2o3(rgbList);
    return image2dHistogram(o1o2o3List, mask);
  }
}
