# Video Special Effects in C++ with OpenCV

A real-time video processing pipeline built in C++ for CS 5330 Computer Vision
at Northeastern University. Implements a set of image filters from scratch and
integrates two deep neural networks (Depth Anything V2 and a fast neural style
transfer model) via the ONNX Runtime — all running on a live webcam stream
controlled by keyboard shortcuts.

## Features

**From-scratch filters** (implemented without `cv::filter2D` or other OpenCV
shortcuts):

- OpenCV-standard and custom grayscale conversions
- Sepia tone with radial vignette
- 5×5 Gaussian blur — both naive 2D and separable 1D versions (3.5× speedup)
- 3×3 Sobel X and Y edge detectors
- Gradient magnitude
- Cartoon (blur + quantize)
- Chromatic aberration (radial RGB shift)
- "Sin City" hue isolation with face-region preservation
- Depth-based exponential fog (Beer-Lambert)

**Deep learning integration:**

- Depth Anything V2 via ONNX Runtime — used for depth visualization
  and depth-aware portrait-mode background blur
- Fast neural style transfer (Van Gogh-style Mosaic model) — second deep
  network in the same pipeline, demonstrating modular ONNX integration

**Face detection:**

- OpenCV Haar cascade for real-time face tracking, used standalone and
  combined with the Sin City color isolation filter

## Tech stack

- C++17, OpenCV 4.x (prebuilt vc16 on Windows)
- ONNX Runtime 1.26 CPU (with Windows-specific patches for `wchar_t` paths
  and the `SimplifiedLayerNormFusion` graph optimization bug)
- CMake build system
- VS Code + MSVC

## Build

Requires OpenCV 4. ONNX Runtime is only required for `vidDisplay`, which uses
the depth and style-transfer networks.

```bash
cmake -S . -B build -DBUILD_ONNX_EFFECTS=OFF
cmake --build build --config Release
```

To build `vidDisplay`, point CMake at an ONNX Runtime install:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
```

The model files (`model.onnx`, `mosaic-9.onnx`, `haarcascade_frontalface_alt2.xml`)
need to be in the working directory at runtime. Pretrained models can be
downloaded from:

- Depth Anything V2: https://huggingface.co/onnx-community
- Fast neural style transfer: https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style

## Controls

Run `vidDisplay.exe` and press keys to toggle effects:

| Key                | Effect                               |
| ------------------ | ------------------------------------ |
| `q`                | Quit                                 |
| `s`                | Save current frame                   |
| `g` / `h`          | OpenCV / custom grayscale            |
| `e`                | Sepia tone                           |
| `b`                | 5×5 Gaussian blur                    |
| `x` / `y` / `m`    | Sobel X, Sobel Y, gradient magnitude |
| `l`                | Blur + quantize (cartoon)            |
| `f`                | Face detection                       |
| `d`                | Depth visualization                  |
| `Shift+F`          | Depth-based portrait blur            |
| `a`                | Chromatic aberration                 |
| `i`, `]`           | Sin City color isolation, cycle hue  |
| `o`, `[`, `,`, `.` | Depth fog, cycle color, density ±    |
| `v`                | Van Gogh style transfer              |

## Screenshots

[Add screenshots here once uploaded — drag images into the README on GitHub
and they'll get embedded automatically.]

## Acknowledgements

Built for Prof. Bruce Maxwell's CS 5330 Computer Vision course at Northeastern.
The `DA2Network.hpp` wrapper, `faceDetect.cpp/h`, and the Haar cascade XML
were provided with the course materials.
