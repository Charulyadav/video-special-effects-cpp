/*
  Charul - 002535330
  15th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  Static image viewer with keypress-toggled filters (Task 1).
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // check that the user passed an image path
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // read the image from disk
    cv::Mat originalImage = cv::imread(argv[1]);
    if (originalImage.empty())
    {
        cout << "Error: could not read image at " << argv[1] << endl;
        return -1;
    }

    // make a working copy so we can modify it without losing the original
    cv::Mat displayImage = originalImage.clone();

    // create a named window
    const string windowName = "Image Display";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cout << "Controls:\n"
         << "q - quit\n"
         << "g - convert to grayscale\n"
         << "c - reset to original (color)\n"
         << "b - apply Gaussian blur\n"
         << "s - save current view to output.jpg\n";

    // main loop: show image and wait for keypress
    while (true)
    {
        cv::imshow(windowName, displayImage);
        int key = cv::waitKey(0); // wait indefinitely for a key

        if (key == 'q')
        {
            break;
        }
        else if (key == 'g')
        {
            cv::Mat gray;
            cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
            // imshow wants 3 channels for normal display, but it auto-handles 1-channel too
            displayImage = gray;
        }
        else if (key == 'c')
        {
            displayImage = originalImage.clone();
        }
        else if (key == 'b')
        {
            cv::GaussianBlur(originalImage, displayImage, cv::Size(15, 15), 0);
        }
        else if (key == 's')
        {
            cv::imwrite("output.jpg", displayImage);
            cout << "Saved to output.jpg\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}