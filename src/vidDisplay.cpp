/*
  Charul - 002535330
  16th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  Main live video application - captures webcam frames and applies toggleable filters and deep-network effects via keypresses (Tasks 2-12 + extension).
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"
#include "StyleNetwork.hpp"

/*
  main() opens the default camera, allocates the DA2 depth network and
  the style transfer network, then enters the capture loop. each iteration:
  grabs a frame, branches on the currently-active mode flag to apply the
  selected filter, displays the result, and checks for a keypress. keys
  toggle mode flags; pressing the same key again turns the effect off.
*/
int main(int argc, char *argv[])
{
    // open the default camera (device 0)
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    // report the frame size the camera is giving us
    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " x " << refS.height << "\n";

    const std::string windowName = "Video";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cv::Mat displayFrame;
    int saveCounter = 0;

    // mode flags (toggled by keypresses)
    bool grayscaleMode = false;
    bool altGrayscaleMode = false;
    bool sepiaMode = false;
    bool blurMode = false;
    bool sobelXMode = false;
    bool sobelYMode = false;
    bool magnitudeMode = false;
    bool quantizeMode = false;
    bool faceDetectMode = false;
    bool depthMode = false;
    bool depthBlurMode = false;
    bool chromAberrationMode = false;
    bool colorIsolationMode = false;
    int isolationHue = 0; // 0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta
    bool fogMode = false;
    float fogDensity = 2.5f;
    int fogColorPreset = 0; // 0=cold, 1=warm, 2=neutral
    bool styleMode = false;

    std::cout << "Controls:\n"
              << "q - quit\n"
              << "s - save current frame to capture_<N>.jpg\n"
              << "g - toggle grayscale (cvtColor)\n"
              << "h - toggle custom grayscale\n"
              << "e - toggle sepia tone\n"
              << "b - toggle 5x5 blur (fast version)\n"
              << "x - toggle Sobel X (vertical edges)\n"
              << "y - toggle Sobel Y (horizontal edges)\n"
              << "m - toggle gradient magnitude\n"
              << "l - toggle blur+quantize (cartoon)\n"
              << "f - toggle face detection\n"
              << "d - toggle depth visualization\n"
              << "F - toggle depth-based background blur\n"
              << "a - toggle chromatic aberration\n"
              << "i - toggle color isolation (Sin City)\n"
              << "] - cycle isolation color\n"
              << "o - toggle depth fog\n"
              << "[ - cycle fog color preset\n"
              << ", - decrease fog density,  . - increase fog density\n"
              << "v - toggle Van Gogh style transfer (mosaic)\n";

    DA2Network da_net("model.onnx");
    StyleNetwork style_net("mosaic-9.onnx");

    // scale_factor sizes the input to ~256px tall - speed/quality balance
    const float reduction = 0.5f;
    float scale_factor = 256.0f / (refS.height * reduction);
    std::cout << "DA2 scale factor: " << scale_factor << "\n";

    for (;;)
    {
        capdev >> frame; // grab next frame from the stream
        if (frame.empty())
        {
            std::cout << "Frame is empty\n";
            break;
        }

        // apply transformations based on current mode
        if (grayscaleMode)
        {
            cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2GRAY);
        }
        else if (altGrayscaleMode)
        {
            greyscale(frame, displayFrame);
        }
        else if (sepiaMode)
        {
            sepia(frame, displayFrame);
        }
        else if (blurMode)
        {
            blur5x5_2(frame, displayFrame);
        }
        else if (sobelXMode)
        {
            cv::Mat sobel;
            sobelX3x3(frame, sobel);
            cv::convertScaleAbs(sobel, displayFrame); //|value|, packed back into uchar
        }
        else if (sobelYMode)
        {
            cv::Mat sobel;
            sobelY3x3(frame, sobel);
            cv::convertScaleAbs(sobel, displayFrame);
        }
        else if (magnitudeMode)
        {
            cv::Mat sx, sy;
            sobelX3x3(frame, sx);
            sobelY3x3(frame, sy);
            magnitude(sx, sy, displayFrame);
        }
        else if (quantizeMode)
        {
            blurQuantize(frame, displayFrame, 10); // default 10 levels
        }
        else if (faceDetectMode)
        {
            // detectFaces needs a grayscale image
            cv::Mat grey;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);

            // start with the color frame, then overlay rectangles
            displayFrame = frame.clone();
            drawBoxes(displayFrame, faces);
        }
        else if (depthMode)
        {
            // resize down for the network (faster)
            cv::Mat smallFrame;
            cv::resize(frame, smallFrame, cv::Size(), reduction, reduction);
            cv::Mat depth;
            da_net.set_input(smallFrame, scale_factor);
            da_net.run_network(depth, smallFrame.size());
            // colorize for nicer visualization
            cv::Mat depthColor;
            cv::applyColorMap(depth, depthColor, cv::COLORMAP_INFERNO);
            // scale back up to original frame size
            cv::resize(depthColor, displayFrame, frame.size());
        }
        else if (depthBlurMode)
        {
            // CREATIVE FILTER: depth-based background blur (portrait mode)
            // foreground stays sharp; background gets blurred.
            cv::Mat smallFrame;
            cv::resize(frame, smallFrame, cv::Size(), reduction, reduction);
            cv::Mat depth;
            da_net.set_input(smallFrame, scale_factor);
            da_net.run_network(depth, smallFrame.size());
            // resize depth back to full frame size
            cv::Mat depthFull;
            cv::resize(depth, depthFull, frame.size());

            // blur a copy of the whole frame
            cv::Mat blurredFull;
            blur5x5_2(frame, blurredFull);

            // per pixel: depth value 0 = close (keep original), 255 = far (use blurred)
            // use a smooth blend so transitions don't look like cutouts
            displayFrame = cv::Mat::zeros(frame.size(), CV_8UC3);
            for (int row = 0; row < frame.rows; row++)
            {
                cv::Vec3b *srcRow = frame.ptr<cv::Vec3b>(row);
                cv::Vec3b *blurRow = blurredFull.ptr<cv::Vec3b>(row);
                uchar *depthRow = depthFull.ptr<uchar>(row);
                cv::Vec3b *outRow = displayFrame.ptr<cv::Vec3b>(row);

                for (int col = 0; col < frame.cols; col++)
                {
                    // alpha = 0 -> all sharp (foreground), alpha = 1 -> all blurred (background)
                    float alpha = 1.0f - (depthRow[col] / 255.0f);
                    for (int c = 0; c < 3; c++)
                    {
                        outRow[col][c] = cv::saturate_cast<uchar>(
                            (1.0f - alpha) * srcRow[col][c] + alpha * blurRow[col][c]);
                    }
                }
            }
        }
        else if (chromAberrationMode)
        {
            chromaticAberration(frame, displayFrame, 15);
        }
        else if (colorIsolationMode)
        {
            // need faces, so detect them in this branch
            cv::Mat grey;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);
            colorIsolation(frame, displayFrame, isolationHue, faces);
        }
        else if (fogMode)
        {
            // get depth map
            cv::Mat smallFrame;
            cv::resize(frame, smallFrame, cv::Size(), reduction, reduction);
            cv::Mat depthSmall;
            da_net.set_input(smallFrame, scale_factor);
            da_net.run_network(depthSmall, smallFrame.size());
            // resize depth back to full frame size
            cv::Mat depthFull;
            cv::resize(depthSmall, depthFull, frame.size());

            // choose fog color from preset
            cv::Vec3b fogColor;
            if (fogColorPreset == 0)
                fogColor = cv::Vec3b(220, 200, 180); // cold
            else if (fogColorPreset == 1)
                fogColor = cv::Vec3b(150, 180, 220); // warm
            else
                fogColor = cv::Vec3b(220, 220, 220); // neutral

            depthFog(frame, depthFull, displayFrame, fogDensity, fogColor);
        }
        else if (styleMode)
        {
            // downscale to keep inference fast on CPU
            cv::Mat smallFrame;
            cv::resize(frame, smallFrame, cv::Size(224, 224));

            // apply style transfer
            cv::Mat stylized;
            try
            {
                style_net.run(smallFrame, stylized);
                cv::resize(stylized, displayFrame, frame.size());
            }
            catch (const Ort::Exception &e)
            {
                std::cout << "Style transfer error: " << e.what() << "\n";
                styleMode = false; // disable so we don't keep crashing
                displayFrame = frame.clone();
            }
        }
        else
        {
            displayFrame = frame.clone();
        }

        cv::imshow(windowName, displayFrame);

        // wait briefly for a key; 10ms keeps the loop ~100 fps cap
        int key = cv::waitKey(10);

        if (key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            std::string filename = "capture_" + std::to_string(saveCounter++) + ".jpg";
            cv::imwrite(filename, displayFrame);
            std::cout << "Saved " << filename << "\n";
        }
        else if (key == 'g')
        {
            grayscaleMode = !grayscaleMode;
            if (grayscaleMode)
                altGrayscaleMode = false;
            std::cout << "Grayscale: " << (grayscaleMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'h')
        {
            altGrayscaleMode = !altGrayscaleMode;
            if (altGrayscaleMode)
                grayscaleMode = false;
            std::cout << "Custom grayscale: " << (altGrayscaleMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'e')
        {
            sepiaMode = !sepiaMode;
            if (sepiaMode)
            {
                grayscaleMode = false;
                altGrayscaleMode = false;
            }
            std::cout << "Sepia: " << (sepiaMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'b')
        {
            blurMode = !blurMode;
            if (blurMode)
            {
                grayscaleMode = false;
                altGrayscaleMode = false;
                sepiaMode = false;
            }
            std::cout << "Blur: " << (blurMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'x')
        {
            sobelXMode = !sobelXMode;
            if (sobelXMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = blurMode = sobelYMode = false;
            }
            std::cout << "Sobel X: " << (sobelXMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'y')
        {
            sobelYMode = !sobelYMode;
            if (sobelYMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = blurMode = sobelXMode = false;
            }
            std::cout << "Sobel Y: " << (sobelYMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'm')
        {
            magnitudeMode = !magnitudeMode;
            if (magnitudeMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
            }
            std::cout << "Magnitude: " << (magnitudeMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'l')
        {
            quantizeMode = !quantizeMode;
            if (quantizeMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = magnitudeMode = false;
            }
            std::cout << "Blur+quantize: " << (quantizeMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'f')
        {
            faceDetectMode = !faceDetectMode;
            if (faceDetectMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = false;
            }
            std::cout << "Face detection: " << (faceDetectMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'd')
        {
            depthMode = !depthMode;
            if (depthMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthBlurMode = false;
            }
            std::cout << "Depth: " << (depthMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'F')
        { // capital F because lowercase f is face detection
            depthBlurMode = !depthBlurMode;
            if (depthBlurMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthMode = false;
            }
            std::cout << "Depth blur: " << (depthBlurMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'a')
        {
            chromAberrationMode = !chromAberrationMode;
            if (chromAberrationMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthMode = depthBlurMode = false;
            }
            std::cout << "Chromatic aberration: " << (chromAberrationMode ? "ON" : "OFF") << "\n";
        }
        else if (key == 'i')
        {
            colorIsolationMode = !colorIsolationMode;
            if (colorIsolationMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthMode = depthBlurMode = chromAberrationMode = false;
            }
            std::cout << "Color isolation: " << (colorIsolationMode ? "ON" : "OFF") << "\n";
        }
        else if (key == ']')
        {
            // cycle through hues: 0(red) -> 30(yel) -> 60(grn) -> 90(cy) -> 120(bl) -> 150(mg)
            isolationHue = (isolationHue + 30) % 180;
            const char *name = "red";
            if (isolationHue == 30)
                name = "yellow";
            else if (isolationHue == 60)
                name = "green";
            else if (isolationHue == 90)
                name = "cyan";
            else if (isolationHue == 120)
                name = "blue";
            else if (isolationHue == 150)
                name = "magenta";
            std::cout << "Isolation color: " << name << "\n";
        }
        else if (key == 'o')
        {
            fogMode = !fogMode;
            if (fogMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthMode = depthBlurMode = chromAberrationMode = false;
                colorIsolationMode = false;
            }
            std::cout << "Depth fog: " << (fogMode ? "ON" : "OFF") << "\n";
        }
        else if (key == '[')
        {
            fogColorPreset = (fogColorPreset + 1) % 3;
            const char *names[] = {"cold (blue)", "warm (golden)", "neutral (white)"};
            std::cout << "Fog color: " << names[fogColorPreset] << "\n";
        }
        else if (key == '.')
        {
            fogDensity = std::min(fogDensity + 0.5f, 10.0f);
            std::cout << "Fog density: " << fogDensity << "\n";
        }
        else if (key == ',')
        {
            fogDensity = std::max(fogDensity - 0.5f, 0.5f);
            std::cout << "Fog density: " << fogDensity << "\n";
        }
        else if (key == 'v')
        {
            styleMode = !styleMode;
            if (styleMode)
            {
                grayscaleMode = altGrayscaleMode = sepiaMode = false;
                blurMode = sobelXMode = sobelYMode = false;
                magnitudeMode = quantizeMode = faceDetectMode = false;
                depthMode = depthBlurMode = chromAberrationMode = false;
                colorIsolationMode = fogMode = false;
            }
            std::cout << "Van Gogh style: " << (styleMode ? "ON" : "OFF") << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}