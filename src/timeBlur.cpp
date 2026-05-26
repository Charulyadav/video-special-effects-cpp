/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.
  (Modified for Windows: replaced sys/time.h with std::chrono)

  Program takes a path to an image on the command line
*/

/*
  Charul - 002535330
  18th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  Timing harness comparing the naive vs. separable 5x5 blur implementations (Task 6). Modified from Prof. Maxwell's original to use std::chrono on Windows.
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "filter.h"

// returns a double which gives time in seconds
double getTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

int main(int argc, char *argv[])
{
    cv::Mat src;
    cv::Mat dst;
    char filename[256];

    if (argc < 2)
    {
        printf("Usage %s <image filename>\n", argv[0]);
        return -1;
    }
    strcpy(filename, argv[1]);

    src = cv::imread(filename);
    if (src.data == NULL)
    {
        printf("Unable to read image %s\n", filename);
        return -1;
    }

    const int Ntimes = 10;

    // set up the timing for version 1
    double startTime = getTime();

    for (int i = 0; i < Ntimes; i++)
    {
        blur5x5_1(src, dst);
    }

    double endTime = getTime();
    double difference = (endTime - startTime) / Ntimes;
    printf("Time per image (1): %.4lf seconds\n", difference);

    // set up the timing for version 2
    startTime = getTime();

    for (int i = 0; i < Ntimes; i++)
    {
        blur5x5_2(src, dst);
    }

    endTime = getTime();
    difference = (endTime - startTime) / Ntimes;
    printf("Time per image (2): %.4lf seconds\n", difference);

    cv::imwrite("cathedral_blurred.jpg", dst);
    printf("Saved cathedral_blurred.jpg\n");

    printf("Terminating\n");

    return 0;
}