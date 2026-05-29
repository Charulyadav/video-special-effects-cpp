/*
  Charul - 002535330
  17th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  From-scratch implementations of grayscale, sepia, blur, Sobel, magnitude, quantize, chromatic aberration, color isolation, and depth fog filters.
*/

#include "filter.h"
#include <algorithm>
#include <cmath>

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
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    // allocate dst as a 3-channel image of the same size as src
    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int row = 0; row < src.rows; row++)
    {
        // get pointers to the start of each row (faster than .at<>())
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            // OpenCV stores pixels as BGR: [0]=B, [1]=G, [2]=R
            uchar B = srcRow[col][0];
            uchar G = srcRow[col][1];

            // custom formula: invert the max of B and G
            uchar value = 255 - std::max(B, G);

            // copy the same value into all three channels so it stays "gray"
            dstRow[col][0] = value;
            dstRow[col][1] = value;
            dstRow[col][2] = value;
        }
    }

    return 0;
}

/* sepia tone filter using a 3x3 channel-mixing matrix with optional radial vignette.
  all three new BGR values are computed from the ORIGINAL three values (read into
  local variables first) to avoid self-contamination. Values are clamped with
  saturate_cast since the red row coefficients sum to >1.

  arguments:
    src - input BGR image (CV_8UC3)
    dst - output sepia-toned image (CV_8UC3) with vignette applied
  returns 0 on success, -1 if src is empty.
*/
int sepia(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            // read all three originals into locals FIRST
            uchar B = srcRow[col][0];
            uchar G = srcRow[col][1];
            uchar R = srcRow[col][2];

            // compute new values using ONLY the locals (originals)
            // coefficients applied as: new_channel = aR + bG + cB
            float newB = 0.272f * R + 0.534f * G + 0.131f * B;
            float newG = 0.349f * R + 0.686f * G + 0.168f * B;
            float newR = 0.393f * R + 0.769f * G + 0.189f * B;

            // vignette: darken based on distance from image center
            float cx = src.cols / 2.0f;
            float cy = src.rows / 2.0f;
            float dx = (col - cx) / cx;
            float dy = (row - cy) / cy;
            float dist = std::sqrt(dx * dx + dy * dy);           // 0 at center, ~1.4 at corners
            float vignette = std::max(0.0f, 1.0f - 0.6f * dist); // tweak 0.6 for strength

            newB *= vignette;
            newG *= vignette;
            newR *= vignette;

            // clamp to [0, 255] and write
            // saturate_cast<uchar> handles both negative and >255 cases
            dstRow[col][0] = cv::saturate_cast<uchar>(newB);
            dstRow[col][1] = cv::saturate_cast<uchar>(newG);
            dstRow[col][2] = cv::saturate_cast<uchar>(newR);
        }
    }

    return 0;
}

/* naive 5x5 Gaussian blur using the full 2D kernel and cv::Mat::at<>() for access.
  kernel: [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1], divisor 100.
  the outer two rows/columns are copied from src unchanged.

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output blurred image (CV_8UC3), same size as src
  returns 0 on success, -1 if src is empty.
*/
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    // start with a copy so the unprocessed border has valid pixel values
    dst = src.clone();

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};
    const int divisor = 100;

    for (int row = 2; row < src.rows - 2; row++)
    {
        for (int col = 2; col < src.cols - 2; col++)
        {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int kr = -2; kr <= 2; kr++)
            {
                for (int kc = -2; kc <= 2; kc++)
                {
                    //.at<>() does bounds checking on every call
                    cv::Vec3b p = src.at<cv::Vec3b>(row + kr, col + kc);
                    int w = kernel[kr + 2][kc + 2];
                    sumB += p[0] * w;
                    sumG += p[1] * w;
                    sumR += p[2] * w;
                }
            }
            dst.at<cv::Vec3b>(row, col)[0] = sumB / divisor;
            dst.at<cv::Vec3b>(row, col)[1] = sumG / divisor;
            dst.at<cv::Vec3b>(row, col)[2] = sumR / divisor;
        }
    }
    return 0;
}

/* fast 5x5 Gaussian blur using two separable 1D passes ([1 2 4 2 1]/10 horizontal
  and vertical) and cv::Mat::ptr<>() for direct row-pointer access. mathematically
  equivalent to blur5x5_1 but ~3.5x faster in Release mode.

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output blurred image (CV_8UC3), same size as src
  returns 0 on success, -1 if src is empty.
*/
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    dst = src.clone();
    cv::Mat temp = src.clone(); // intermediate buffer for horizontal pass result

    // horizontal pass: src -> temp
    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(row);

        for (int col = 2; col < src.cols - 2; col++)
        {
            // unrolled: apply [1 2 4 2 1] / 10 across the 5 neighbors
            for (int c = 0; c < 3; c++)
            {
                tempRow[col][c] = (srcRow[col - 2][c] + 2 * srcRow[col - 1][c] + 4 * srcRow[col][c] + 2 * srcRow[col + 1][c] + srcRow[col + 2][c]) / 10;
            }
        }
    }

    // vertical pass: temp -> dst
    for (int row = 2; row < src.rows - 2; row++)
    {
        // grab pointers to the 5 rows we'll combine
        cv::Vec3b *rm2 = temp.ptr<cv::Vec3b>(row - 2);
        cv::Vec3b *rm1 = temp.ptr<cv::Vec3b>(row - 1);
        cv::Vec3b *r0 = temp.ptr<cv::Vec3b>(row);
        cv::Vec3b *rp1 = temp.ptr<cv::Vec3b>(row + 1);
        cv::Vec3b *rp2 = temp.ptr<cv::Vec3b>(row + 2);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                dstRow[col][c] = (rm2[col][c] + 2 * rm1[col][c] + 4 * r0[col][c] + 2 * rp1[col][c] + rp2[col][c]) / 10;
            }
        }
    }

    return 0;
}

/* 3x3 Sobel X filter, implemented as two separable 1D passes:
  vertical smoothing [1 2 1]^T followed by horizontal difference [-1 0 1].
  positive right (bright-on-right gradient gives positive value).

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output signed-short image (CV_16SC3); values in [-255, 255]
  returns 0 on success, -1 if src is empty.
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    // intermediate buffer holds the result of the first pass (still signed)
    cv::Mat temp(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // vertical smoothing pass: [1 2 1]^T, src -> temp
    // reads 3 rows of uchar, writes a row of short
    for (int row = 1; row < src.rows - 1; row++)
    {
        cv::Vec3b *rm1 = src.ptr<cv::Vec3b>(row - 1);
        cv::Vec3b *r0 = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *rp1 = src.ptr<cv::Vec3b>(row + 1);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);

        for (int col = 0; col < src.cols; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                tempRow[col][c] = (rm1[col][c] + 2 * r0[col][c] + rp1[col][c]) / 4;
            }
        }
    }

    // horizontal difference pass: [-1 0 1], temp -> dst
    // reads 3 cols of short, writes a short
    for (int row = 1; row < src.rows - 1; row++)
    {
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(row);

        for (int col = 1; col < src.cols - 1; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                // positive right: right-pixel minus left-pixel
                dstRow[col][c] = (-1 * tempRow[col - 1][c] + 0 + 1 * tempRow[col + 1][c]) / 2;
            }
        }
    }

    return 0;
}

/* 3x3 Sobel Y filter, implemented as two separable 1D passes:
  vertical difference [1 0 -1]^T followed by horizontal smoothing [1 2 1].
  positive up (the sign is flipped from the standard textbook kernel so that
  brightening upward in screen coordinates gives positive values).

  arguments:
    src - input BGR image (CV_8UC3); not modified
    dst - output signed-short image (CV_16SC3); values in [-255, 255]
  returns 0 on success, -1 if src is empty.
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;

    cv::Mat temp(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // vertical difference pass: [1 0 -1]^T, src -> temp
    // positive UP: row-above minus row-below (since row 0 is at the top of screen)
    for (int row = 1; row < src.rows - 1; row++)
    {
        cv::Vec3b *rm1 = src.ptr<cv::Vec3b>(row - 1);
        cv::Vec3b *rp1 = src.ptr<cv::Vec3b>(row + 1);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);

        for (int col = 0; col < src.cols; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                tempRow[col][c] = (rm1[col][c] - rp1[col][c]) / 2;
            }
        }
    }

    // horizontal smoothing pass: [1 2 1], temp -> dst
    for (int row = 1; row < src.rows - 1; row++)
    {
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(row);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(row);

        for (int col = 1; col < src.cols - 1; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                dstRow[col][c] = (tempRow[col - 1][c] + 2 * tempRow[col][c] + tempRow[col + 1][c]) / 4;
            }
        }
    }

    return 0;
}

/* gradient magnitude from X and Y Sobel images.
  computes I = sqrt(sx^2 + sy^2) per channel and clamps to [0, 255].
  inputs are cast to float before squaring to avoid short overflow.

  arguments:
    sx  - X-direction Sobel output (CV_16SC3)
    sy  - Y-direction Sobel output (CV_16SC3); must match sx in size
    dst - output magnitude image (CV_8UC3), ready for imshow
  returns 0 on success, -1 if inputs are empty or sizes mismatch.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    if (sx.empty() || sy.empty())
        return -1;
    if (sx.size() != sy.size())
        return -1;

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    for (int row = 0; row < sx.rows; row++)
    {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(row);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < sx.cols; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                // cast to float for the multiplication to avoid short overflow
                float gx = sxRow[col][c];
                float gy = syRow[col][c];
                float mag = std::sqrt(gx * gx + gy * gy);
                // saturate_cast clamps to [0, 255] and rounds
                dstRow[col][c] = cv::saturate_cast<uchar>(mag * 1.5f);
            }
        }
    }

    return 0;
}

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
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    if (src.empty() || levels <= 0 || levels > 255)
        return -1;

    // blur the image (reuse our fast separable blur)
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    // quantize each channel
    dst = cv::Mat::zeros(blurred.size(), CV_8UC3);
    int b = 255 / levels;

    for (int row = 0; row < blurred.rows; row++)
    {
        cv::Vec3b *srcRow = blurred.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < blurred.cols; col++)
        {
            for (int c = 0; c < 3; c++)
            {
                int x = srcRow[col][c];
                int xt = x / b;  // which bucket
                int xf = xt * b; // bucket's lower edge
                dstRow[col][c] = (uchar)xf;
            }
        }
    }

    return 0;
}

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
int chromaticAberration(cv::Mat &src, cv::Mat &dst, int shift)
{
    if (src.empty())
        return -1;

    dst = src.clone(); // start as a copy; green channel stays put

    float cx = src.cols / 2.0f;
    float cy = src.rows / 2.0f;
    // normalize so corner distance = 1.0
    float maxDist = std::sqrt(cx * cx + cy * cy);

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            // distance from center, normalized to [0, 1]
            float dx = col - cx;
            float dy = row - cy;
            float dist = std::sqrt(dx * dx + dy * dy) / maxDist;

            // scale shift radially - 0 at center, full at corners
            // square the distance for a more dramatic edge falloff
            int radialShift = (int)(shift * dist * dist);

            // direction of shift: along the line from center outward
            //(this is what "real" lens aberration looks like)
            float dirX = (dist > 0.001f) ? dx / (dist * maxDist) : 0;
            float dirY = (dist > 0.001f) ? dy / (dist * maxDist) : 0;

            // red channel: sample from offset position (outward)
            int rCol = col + (int)(radialShift * dirX);
            int rRow = row + (int)(radialShift * dirY);
            // blue channel: sample from opposite direction (inward)
            int bCol = col - (int)(radialShift * dirX);
            int bRow = row - (int)(radialShift * dirY);

            // clamp to image bounds
            rCol = std::max(0, std::min(src.cols - 1, rCol));
            rRow = std::max(0, std::min(src.rows - 1, rRow));
            bCol = std::max(0, std::min(src.cols - 1, bCol));
            bRow = std::max(0, std::min(src.rows - 1, bRow));

            // sample R from one location, B from another, keep G in place
            dstRow[col][0] = src.at<cv::Vec3b>(bRow, bCol)[0]; // B
            dstRow[col][1] = srcRow[col][1];                   // G (unchanged)
            dstRow[col][2] = src.at<cv::Vec3b>(rRow, rCol)[2]; // R
        }
    }

    return 0;
}

/* Sin City color isolation. keeps pixels close to targetHue (in HSV) in full
  color and the detected face regions in full color; everything else converts
  to grayscale.

  algorithm steps:
    1. convert src to HSV.
    2. build a 3-channel grayscale version of src.
    3. per-pixel: if hue is within +/- 15 of targetHue AND saturation > 70,
       keep original color; otherwise use grayscale.
    4. paint the face rectangles back in full color.

  arguments:
    src        - input BGR image (CV_8UC3)
    dst        - output isolated-color image (CV_8UC3)
    targetHue  - target hue on OpenCV's HSV scale [0..179]
                 (0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta)
    faces      - vector of face rectangles from detectFaces()
  returns 0 on success, -1 if src is empty.
*/
int colorIsolation(cv::Mat &src, cv::Mat &dst, int targetHue,
                   std::vector<cv::Rect> &faces)
{
    if (src.empty())
        return -1;

    // convert to HSV so we can isolate by hue
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // build a grayscale version of the frame (as 3 channels)
    cv::Mat gray, grayBGR;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR); // back to 3 channels

    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    const int hueTolerance = 15; // how close to the target hue counts
    const int satThreshold = 70; // minimum saturation (filters out gray)

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *hsvRow = hsv.ptr<cv::Vec3b>(row);
        cv::Vec3b *grayRow = grayBGR.ptr<cv::Vec3b>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            uchar h = hsvRow[col][0];
            uchar s = hsvRow[col][1];

            // hue is circular - distance from target wraps around at 180
            int hueDiff = std::abs((int)h - targetHue);
            if (hueDiff > 90)
                hueDiff = 180 - hueDiff; // wrap-around

            bool isTargetColor = (hueDiff < hueTolerance) && (s > satThreshold);

            if (isTargetColor)
            {
                dstRow[col] = srcRow[col]; // keep original color
            }
            else
            {
                dstRow[col] = grayRow[col]; // grayscale
            }
        }
    }

    // paint the face regions back in full color
    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Rect f = faces[i];
        // clamp the rect to the image bounds (face boxes can extend off-frame)
        f.x = std::max(0, f.x);
        f.y = std::max(0, f.y);
        f.width = std::min(f.width, src.cols - f.x);
        f.height = std::min(f.height, src.rows - f.y);
        // copy that region from src directly into dst
        src(f).copyTo(dst(f));
    }

    return 0;
}

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
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst,
             float density, cv::Vec3b fogColor)
{
    if (src.empty() || depth.empty())
        return -1;
    if (src.size() != depth.size())
        return -1;

    dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(row);
        uchar *depthRow = depth.ptr<uchar>(row);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            // normalize depth to [0, 1]
            float d = 1.0f - (depthRow[col] / 255.0f);

            // Beer-Lambert transmission: fraction of original light that reaches camera
            float transmission = std::exp(-density * d);
            float fog = 1.0f - transmission;

            for (int c = 0; c < 3; c++)
            {
                float blended = srcRow[col][c] * transmission + fogColor[c] * fog;
                dstRow[col][c] = cv::saturate_cast<uchar>(blended);
            }
        }
    }

    return 0;
}
