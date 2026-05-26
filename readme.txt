==========================================================================
CS 5330 - Project 1: Video Special Effects
==========================================================================

NAME
    Charul - 002535330

GROUP MEMBERS
    None - solo submission.

==========================================================================
DEVELOPMENT SETUP
==========================================================================

OS:          Windows
IDE/Editor:  Visual Studio Code
Compiler:    MSVC (Visual Studio Build Tools 2026)
Build:       CMake
OpenCV:      4.x (prebuilt for Windows, vc16)
ONNX Runtime: CPU build, linked against vidDisplay for DA2 and style transfer

==========================================================================
EXECUTABLES BUILT
==========================================================================

bin/Debug or bin/Release contain:

  imgDisplay.exe   — Task 1 (still-image viewer with keypress filters)
  vidDisplay.exe   — Tasks 2-12 + extension (main video application)
  timeBlur.exe     — Task 6 timing harness (run on cathedral.jpeg)
  showFaces.exe    — Prof's standalone face-detection demo (reference)

==========================================================================
HOW TO RUN
==========================================================================

From the project root directory:

    bin\Debug\vidDisplay.exe

Required files in project root for vidDisplay to work:
  - haarcascade_frontalface_alt2.xml   (face detection)
  - model.onnx                         (Depth Anything V2)
  - mosaic-9.onnx                      (style transfer for extension)
  - onnxruntime.dll                    (next to the .exe in bin/...)

==========================================================================
KEYPRESS CONTROLS (vidDisplay)
==========================================================================

  q       Quit
  s       Save current frame as capture_<N>.jpg

  g       OpenCV grayscale (BT.601)
  h       Custom grayscale (255 - max(B, G))
  e       Sepia tone (with vignette)
  b       5x5 Gaussian blur (separable, from Task 6)
  x       Sobel X (vertical edges)
  y       Sobel Y (horizontal edges)
  m       Gradient magnitude
  l       Blur + quantize (cartoon)
  f       Face detection (Haar cascade)
  d       Depth visualization (DA2 + INFERNO colormap)
  Shift+F Depth-based portrait blur
  a       Chromatic aberration
  i       Sin City color isolation
  ]       Cycle isolation hue (red -> yellow -> green -> cyan -> blue -> magenta)
  o       Depth-based exponential fog
  [       Cycle fog color (cold blue / warm gold / neutral white)
  ,  .    Decrease / increase fog density
  v       Van Gogh style transfer (extension)

==========================================================================
EXTENSION
==========================================================================

Neural style transfer using the Mosaic model from the ONNX Model Zoo,
integrated via a new wrapper class StyleNetwork.hpp modeled on the
provided DA2Network.hpp. Toggle with 'v'. Details in the report.

==========================================================================
TIME TRAVEL DAYS
==========================================================================

Not used

==========================================================================
NOTES
==========================================================================

- Build was done in both Debug and Release; Release was used for the
  Task 6 timing numbers.
- The provided model_fp16.onnx triggered a SimplifiedLayerNormFusion
  crash in recent ONNX Runtime versions on Windows, so I switched to a
  standard (non-fp16) Depth Anything V2 ONNX export instead. Loaded as
  "model.onnx" in the project root.
- Two Windows-specific patches were applied to DA2Network.hpp:
    * Ort::Session path conversion to wchar_t inside #ifdef _WIN32
    * Graph optimization set to ORT_ENABLE_BASIC as a safety measure
      since the fusion bug can affect non-fp16 models too on this
      ORT version.
- The same patches were applied to my StyleNetwork.hpp wrapper for the
  style transfer extension.

==========================================================================