/*
  Charul - 002535330
  17th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  Function prototypes for image-processing filters implemented in filter.cpp.
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>

/* custom grayscale conversion.
  takes the max of the B and G channels and inverts it: value = 255 - max(B, G).
  the red channel is ignored. Produces a 3-channel image where each channel
  carries the same value - visually distinct from the BT.601-weighted OpenCV
  conversion because red-heavy pixels become bright instead of mid-tone.

  arguments:
    src - input BGR image (CV_8UC3)
    dst - output 3-channel grayscale image (CV_8UC3)
  returns 0 on success, -1 if src is empty.
*/
int greyscale(cv::Mat &src, cv::Mat &dst);

/* sepia tone filter using a 3x3 channel-mixing matrix with optional radial vignette.
  all three new BGR values are computed from the ORIGINAL three values (read into
  local variables first) to avoid self-contamination. Values are clamped with
  saturate_cast since the red row coefficients sum to >1.

  arguments:
    src - input BGR image (CV_8UC3)
    dst - output sepia-toned image (CV_8UC3) with vignette applied
  returns 0 on success, -1 if src is empty.
*/
int sepia(cv::Mat &src, cv::Mat &dst);

/* naive 5x5 Gaussian blur using the full 2D kernel and cv::Mat::at<>() for access.
  kernel: [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1], divisor 100.
  the outer two rows/columns are copied from src unchanged.

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output blurred image (CV_8UC3), same size as src
  returns 0 on success, -1 if src is empty.
*/
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/* fast 5x5 Gaussian blur using two separable 1D passes ([1 2 4 2 1]/10 horizontal
  and vertical) and cv::Mat::ptr<>() for direct row-pointer access. mathematically
  equivalent to blur5x5_1 but ~3.5x faster in Release mode.

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output blurred image (CV_8UC3), same size as src
  returns 0 on success, -1 if src is empty.
*/
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/* 3x3 Sobel X filter, implemented as two separable 1D passes:
  vertical smoothing [1 2 1]^T followed by horizontal difference [-1 0 1].
  positive right (bright-on-right gradient gives positive value).

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output signed-short image (CV_16SC3); values in [-255, 255]
  returns 0 on success, -1 if src is empty.
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/* 3x3 Sobel Y filter, implemented as two separable 1D passes:
  vertical difference [1 0 -1]^T followed by horizontal smoothing [1 2 1].
  positive up (the sign is flipped from the standard textbook kernel so that
  brightening upward in screen coordinates gives positive values).

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output signed-short image (CV_16SC3); values in [-255, 255]
  returns 0 on success, -1 if src is empty.
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/* gradient magnitude from X and Y Sobel images.
  computes I = sqrt(sx^2 + sy^2) per channel and clamps to [0, 255].
  inputs are cast to float before squaring to avoid short overflow.

  arguments:
    sx  - X-direction Sobel output (CV_16SC3)
    sy  - Y-direction Sobel output (CV_16SC3); must match sx in size
    dst - output magnitude image (CV_8UC3), ready for imshow
  returns 0 on success, -1 if inputs are empty or sizes mismatch.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/* blur the image (using blur5x5_2) and then quantize each color channel
  into 'levels' discrete values per channel. bucket size b = 255/levels;
  each value x becomes (x/b)*b using integer division.
  with levels=10, the image has only 1000 possible colors (vs. 16.7M).

  arguments:
    src    - input BGR image (CV_8UC3); not modified
    dst    - output blurred-and-quantized image (CV_8UC3)
    levels - number of discrete levels per channel (default 10)
  returns 0 on success, -1 if src is empty or levels <= 0.
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/* radial chromatic aberration. shifts the red channel outward along the
  radial direction from image center and the blue channel inward by the
  same amount. shift magnitude grows with the square of distance from
  center, so the center stays sharp and the corners distort. green stays.

  arguments:
    src   - input BGR image (CV_8UC3); not modified
    dst   - output image (CV_8UC3) with chromatic shift applied
    shift - maximum shift in pixels (at the image corners)
  returns 0 on success, -1 if src is empty.
*/
int chromaticAberration(cv::Mat &src, cv::Mat &dst, int shift);

/* Sin City color isolation. keeps pixels close to targetHue (in HSV) in full
  color and the detected face regions in full color; everything else converts
  to grayscale.

  arguments:
    src        - input BGR image (CV_8UC3)
    dst        - output isolated-color image (CV_8UC3)
    targetHue  - target hue on OpenCV's HSV scale [0..179]
                 (0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta)
    faces      - vector of face rectangles from detectFaces()
  returns 0 on success, -1 if src is empty.
*/
int colorIsolation(cv::Mat &src, cv::Mat &dst, int targetHue, std::vector<cv::Rect> &faces);

/* depth-based exponential fog following Beer-Lambert atmospheric scattering.
  for each pixel: output = original * exp(-density*d) + fogColor * (1 - exp(-density*d))
  where d is depth in [0,1] (inverted from DA2's output so 0=close, 1=far).
  exponential falloff matches real-world light scattering and feels more
  natural than a linear fade.

  arguments:
    src       - input BGR image (CV_8UC3); not modified
    depth     - single-channel depth map (CV_8UC1), same size as src
    dst       - output foggy image (CV_8UC3)
    density   - fog accumulation rate (1.0 = light haze, 5.0+ = heavy fog)
    fogColor  - color of the fog in BGR
  returns 0 on success, -1 if inputs are empty or sizes mismatch.
*/
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, float density = 2.5f, cv::Vec3b fogColor = cv::Vec3b(200, 180, 160));

#endif
