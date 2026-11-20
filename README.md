# cpp-tensorrt-inference

Real-time [RF-DETR](https://github.com/roboflow/rf-detr) object detection in C++ with TensorRT on NVIDIA Jetson Orin.

https://github.com/user-attachments/assets/d056b45c-a0d9-421f-a8cf-4a33ce3d7887

Footage: [Dubang Chang on Pexels](https://www.pexels.com/video/busy-urban-street-scene-in-oakland-california-28775145/) (Pexels License), 1920x1080 version

## Overview

This project runs the COCO-pretrained RF-DETR object detection models (nano, small, medium, large) as TensorRT
engines from C++. At runtime the only dependencies are TensorRT, the CUDA runtime, OpenCV and yaml-cpp.

Python is only used offline, to export the model to ONNX and analyze benchmark data. 

Each image frame goes through:

```
frame → preprocess (CPU) → H2D copy → TensorRT inference → D2H copy → postprocess (CPU) → detections
```

- **Preprocess:** resize to the engine's input size, ImageNet normalization, BGR-RGB, HWC-NCHW.
- **Postprocess:** per-query best class, confidence threshold, normalized cxcywh boxes scaled
  to the source frame as xyxy.

Implementation details:

- Input size, query count and class count are read from the engine.
- Per-stage timing: CPU wall-clock for each stage, with CUDA events for the copies and inference.

## Benchmarks

A benchmark executable and sweep script are provided in `benchmark/`. 

Full results, including per-stage p95/p99, can be found here: [`benchmark/results/2026-11-20_1521/`](benchmark/results/2026-11-20_1521/)

### Benchmark Environment
- Jetson AGX Orin Developer Kit
- MAXN power mode
- `jetson_clocks` enabled
- JetPack 6.2 (L4T R36.5)
- TensorRT 10.3
- Batch size 1
- 1920x1080 input video
- 5000 measured frames after 100 warm-up frames per engine
- 60 seconds of cooldown time between runs

No thermal throttling was observed, with a max GPU temperature of 65 °C during runs

FPS is detector throughput (`1000 / mean latency`). This includes preprocessing, inference and
postprocessing, but not video decoding and display.

| Model | Input | Precision | FPS | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Detections/frame |
|---|---|---|---|---|---|---|---|---|
| nano | 384 | FP16 | 153.0 | 6.54 | 6.52 | 6.61 | 6.78 | 22.4 |
| nano | 384 | FP32 | 87.8 | 11.38 | 11.34 | 11.45 | 12.26 | 22.4 |
| small | 512 | FP16 | 102.3 | 9.78 | 9.44 | 11.95 | 13.19 | 23.7 |
| small | 512 | FP32 | 56.9 | 17.58 | 17.40 | 19.73 | 20.21 | 23.7 |
| medium | 576 | FP16 | 90.2 | 11.09 | 10.92 | 13.45 | 13.70 | 25.6 |
| medium | 576 | FP32 | 43.2 | 23.17 | 23.02 | 25.44 | 25.85 | 25.6 |
| large | 704 | FP16 | 54.4 | 18.40 | 18.34 | 18.72 | 19.12 | 27.8 |
| large | 704 | FP32 | 26.3 | 38.03 | 37.96 | 38.34 | 38.73 | 27.9 |

Median per-stage time (ms):

| Model | Precision | Preprocess | H2D | TensorRT | D2H | Postprocess |
|---|---|---|---|---|---|---|
| nano | FP16 | 1.93 | 0.22 | 4.03 | 0.08 | 0.20 |
| large | FP16 | 6.37 | 0.60 | 10.34 | 0.09 | 0.20 |

CPU preprocessing accounts for ~30-35% of frame time, which makes GPU preprocessing the biggest optimization on the roadmap. 

FP16 and FP32 engines produce the same number of detections
per frame on this video, but accuracy (mAP) has not been validated yet.


## Getting started

### Requirements

- Jetson Orin with JetPack 6.2 (CUDA 12.6, TensorRT 10.3 and OpenCV 4.8)
- CMake 3.17+ and yaml-cpp: `sudo apt install cmake libyaml-cpp-dev`
- `trtexec` on your `PATH`: `export PATH="/usr/src/tensorrt/bin:$PATH"`

### 1. Export a TensorRT engine

```bash
python3 -m venv models/.venv
source models/.venv/bin/activate
pip install -r models/requirements.txt

python models/export_rfdetr_tensorrt.py --size nano --precision fp16
```

This downloads the pretrained weights, exports `models/output/rfdetr-nano.onnx`, and builds
`models/output/rfdetr-nano-fp16.engine` with `trtexec`. 

Model sizes are `nano`, `small`, `medium` and `large`. Precisions are `fp16` and `fp32`.

The engine is built with `trtexec` instead of the Python `rfdetr` package TensorRT export, as a
TensorRT engine has to match the TensorRT the C++ binary links against. The TensorRT and CUDA wheels that 
`pip` installs alongside `rfdetr` are newer and aren't compatible with TensorRT 10.3 on JetPack 6.2.

Existing exported ONNX files are reused. Delete them to re-export, for example after upgrading `rfdetr`.

### 2. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. Run

Download the [clip](https://www.pexels.com/video/busy-urban-street-scene-in-oakland-california-28775145/) at 1920x1080 resolution into `data/pexels-oakland-street.mp4`.

Verify the checksum is `7d48d07b436a7d05808db98b24bdc7f2d388359ec7b8d255dbcb64370beaac53` with `sha256sum data/pexels-oakland-street.mp4`.

Set `engine_path` and `input_source` in [`config/default.yaml`](config/default.yaml), then:

```bash
./build/orin_detector config/default.yaml
```

Relative paths in the config are resolved against the config file's directory. Class names come from 
[`config/coco_classes.yaml`](config/coco_classes.yaml), which is
generated by `models/export_rfdetr_pretrained_classes.py`. The keys are COCO category IDs
(1–90, with gaps), which match the pretrained `rfdetr` model weights' output class indices.

### 4. Benchmark

```bash
sudo jetson_clocks
python3 benchmark/benchmark.py              # only runs on existing engines in models/output/
```

Results are saved to `benchmark/results/<timestamp>/`. 

To benchmark a single engine directly, modify `config/default.yaml` with the desired engine, and then run:

```bash
./build/orin_benchmark config/default.yaml <num_frames> <warmup_frames> <out.csv>
```

## Project layout

```
include/, src/
  infer/        TRTInfer: engine loading, GPU buffers, CUDA stream, inference
  models/       RFDETRModel pre/postprocessing
  utils/        config and class-name loading, CUDA_CHECK, ScopedTimer
  main.cpp      demo: runs the detector on a video and draws the detections
benchmark/      headless benchmark executable, benchmark sweep script, results
models/         ONNX/engine and class-name export scripts
config/         default config and COCO class names
```

## Roadmap

- Top-K postprocessing over (query, class) pairs, matching the reference implementation
- RAII wrappers for CUDA resources
- Unit tests and CI
- CUDA preprocessing kernel
- Pinned and zero-copy memory 
- ROS 2 node and end-to-end camera-to-detection latency
- mAP validation against the Python reference

## Related work

- [roboflow/rf-detr](https://github.com/roboflow/rf-detr): the model, training code and Python
  inference

## License

MIT, see [LICENSE](LICENSE). RF-DETR is developed by Roboflow; check the
[rf-detr repository](https://github.com/roboflow/rf-detr) for the license of the code and of
each pretrained model.
