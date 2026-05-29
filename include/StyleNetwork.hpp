/*
  Charul - 002535330
  21th May 2026
  CS 5330 Computer Vision
  Project 1: Video Special Effects

  It is a wrapper for fast-neural-style ONNX models from the
  ONNX Model Zoo. Modeled on Bruce Maxwell's DA2Network.hpp wrapper.

  Usage:
    StyleNetwork net("mosaic-9.onnx");
    cv::Mat output;
    net.run(input_bgr, output);

  The model is a fully-convolutional network - input can be any HxW.
  Smaller input = faster inference. 256-tall inputs run reasonably on CPU.
*/
#ifndef STYLE_NETWORK_HPP
#define STYLE_NETWORK_HPP

#include <cstdio>
#include <cstring>
#include <string>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class StyleNetwork
{
public:
    StyleNetwork(const char *network_path)
    {
        // configure session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

#ifdef _WIN32
        std::wstring wpath(network_path, network_path + std::strlen(network_path));
        this->session_ = new Ort::Session(env, wpath.c_str(), session_options);
#else
        this->session_ = new Ort::Session(env, network_path, session_options);
#endif
    }

    StyleNetwork(const StyleNetwork &) = delete;
    StyleNetwork &operator=(const StyleNetwork &) = delete;

    ~StyleNetwork()
    {
        if (input_data != NULL)
            delete[] input_data;
        delete session_;
    }

    // apply the style transfer to a BGR image.
    // output is the same size and type as the input.
    int run(const cv::Mat &src, cv::Mat &dst)
    {
        if (src.empty())
            return -1;

        int H = src.rows;
        int W = src.cols;

        // allocate input buffer if size changed
        if (H != height_ || W != width_)
        {
            height_ = H;
            width_ = W;
            if (input_data != NULL)
                delete[] input_data;
            input_data = new float[3 * H * W];
            input_shape_[2] = H;
            input_shape_[3] = W;
        }

        // preprocess: BGR uchar [0,255] -> planar RGB float32 [0,255]
        // network expects [R-plane, G-plane, B-plane]
        const int image_size = H * W;
        for (int i = 0; i < H; i++)
        {
            const cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
            float *rPlane = &input_data[i * W];
            float *gPlane = &input_data[image_size + i * W];
            float *bPlane = &input_data[image_size * 2 + i * W];
            for (int j = 0; j < W; j++)
            {
                // OpenCV stores BGR; model wants RGB planes
                rPlane[j] = (float)ptr[j][2]; // R
                gPlane[j] = (float)ptr[j][1]; // G
                bPlane[j] = (float)ptr[j][0]; // B
            }
        }

        // wrap input_data in an ONNX tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data, 3 * H * W,
            input_shape_.data(), input_shape_.size());

        // model layer names from the ONNX zoo fast_neural_style models
        const char *input_names[] = {"input1"};
        const char *output_names[] = {"output1"};
        Ort::RunOptions run_options;
        auto outputs = session_->Run(run_options, input_names, &input_tensor, 1,
                                     output_names, 1);

        // postprocess: planar RGB float -> interleaved BGR uchar, clipped to [0,255]
        const float *out = outputs[0].GetTensorData<float>();
        dst = cv::Mat::zeros(H, W, CV_8UC3);
        for (int i = 0; i < H; i++)
        {
            cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
            const float *rPlane = &out[i * W];
            const float *gPlane = &out[image_size + i * W];
            const float *bPlane = &out[image_size * 2 + i * W];
            for (int j = 0; j < W; j++)
            {
                ptr[j][0] = cv::saturate_cast<uchar>(bPlane[j]); // B
                ptr[j][1] = cv::saturate_cast<uchar>(gPlane[j]); // G
                ptr[j][2] = cv::saturate_cast<uchar>(rPlane[j]); // R
            }
        }
        return 0;
    }

private:
    Ort::Env env;
    Ort::Session *session_ = nullptr;
    float *input_data = nullptr;
    int height_ = 0;
    int width_ = 0;
    std::array<int64_t, 4> input_shape_{1, 3, 0, 0};
};

#endif
