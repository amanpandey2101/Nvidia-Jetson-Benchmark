# Jetson Orin NX (reComputer J4012) TensorRT Load Tester with UCB Alerts

A production-grade Python load testing and benchmarking tool designed for the **NVIDIA Jetson Orin NX (reComputer J4012)** to evaluate object detection throughput and latency. The utility includes a faithful Python implementation of the **LiveSitter UCB (Unusual Behavior) Alert Engine**, matching the simplified rolling-window rules used for the recamera edge.

---

## Key Features

*   **Version-Agnostic TensorRT Engine**: Supports both TensorRT 8.x (JetPack 5.x) and TensorRT 10.x (JetPack 6.x) Python execution context API calls.
*   **Zero-Dependency CUDA Memory Allocation**: Uses standard Python `ctypes` to bind directly to `libcudart` to allocate pinned host memory (`cudaHostAlloc`) and device memory (`cudaMalloc`). Avoids the compilation and version compatibility errors typical of `PyCUDA`.
*   **Zero Disk I/O Bottleneck**: Preloads and pre-processes images into RAM as NumPy arrays before starting, isolating the benchmark to measure pure GPU, CUDA, and CPU throughput.
*   **Multi-Stream Parallelism**: Simulates independent camera feeds using dedicated threads, each with its own TensorRT Execution Context, CUDA stream, and rolling-window `UCBEngine` instance.
*   **Open-Loop Request Generator**: Benchmarks capacity under load via `--target-rps` per stream to test queue growth and latency spikes without feedback-loop delays.
*   **Automatic Stream Scaling Benchmark**: Auto-tests streams (`1, 2, 4, 8, 16, 32, 64`) and identifies saturation limits. Automatically triggers stop cutoffs when CPU/GPU exceeds 95% utilization, RAM exceeds 90%, or P95 latency exceeds 150ms.
*   **tegrastats Subprocess Parser**: Runs `tegrastats` as a background process to collect per-core CPU loads, GPU utility, frequencies, power consumption, and thermal zones.

---

## Installation

### 1. Jetson Orin NX Setup
Ensure that JetPack, CUDA, and TensorRT are installed on your device.

```bash
# Clone or copy this directory to the device
cd livesitter_recomputer/

# Install python dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
Make sure your CSV data file containing URLs is present (defaults to `UCB-DATA.csv`). The script will automatically download and cache images before running.

---

## CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `model.engine` | Path to prebuilt TensorRT engine file |
| `--model-type` | `str` | `yolov5s` | Model post-processor decoder: `yolov5s` or `rtdetr` |
| `--csv-path` | `str` | `UCB-DATA.csv` | CSV dataset containing image URLs |
| `--image-dir` | `str` | `images` | Directory where downloaded images are cached |
| `--streams` | `int` | `1` | Number of simultaneous camera feed streams to test |
| `--duration` | `int` | `60` | Run benchmark for N seconds |
| `--images-per-stream` | `int` | `None` | Run benchmark until each stream finishes N images |
| `--target-rps` | `float` | `None` | Target request rate (FPS) per stream (Open-loop pacing) |
| `--auto-scale` | `flag` | `False` | Run automatic stream scaling benchmark iteration |
| `--max-download` | `int` | `None` | Maximum number of images to cache/target from the CSV (defaults to all URLs in CSV) |
| `--disable-ucb` | `flag` | `False` | Disable UCB alert business logic (run raw inference only) |
| `--min-frames` | `int` | `35` | Minimum frames in window required to trigger alert |
| `--save-detections` | `flag` | `False` | Save detailed bounding boxes and confidence scores to disk frame-by-frame |
| `--preload-limit` | `int` | `250` | Maximum number of preprocessed images to load in RAM |
| `--output-dir` | `str` | `reports` | Directory where performance reports/plots will be written |

---

## Execution Examples


### 1. Standard Single-Run Benchmark (on Jetson Device)
```bash
python benchmark.py --model yolov5s.engine --model-type yolov5s --streams 8 --duration 300
```

### 2. Open-loop RPS Capacity Testing
```bash
python benchmark.py --model rtdetr.engine --model-type rtdetr --streams 4 --target-rps 15 --duration 120
```

### 3. Automatic Stream Scaling Capacity Run
```bash
python benchmark.py --model yolov5s.engine --model-type yolov5s --auto-scale
```

---

## Output Reports

All results are written to the directory specified by `--output-dir` (default: `./reports/`):

1.  **Console Summary**: Displays global FPS, throughput, P95/P99 latency distribution, context inference latency, and hardware averages.
2.  **`benchmark_results.csv`**: Time-series log capturing FPS, latency, GPU load, RAM, and temperatures every second.
3.  **`benchmark_report.json`**: Comprehensive report containing full hardware specifications, run config settings, per-stream stats, and raw latency data points.
4.  **Matplotlib Charts (`.png`)**:
    *   `fps_over_time.png` - Throughput stability.
    *   `latency_over_time.png` - Latency jitter and raw sample distribution.
    *   `gpu_utilization.png` - GPU load percentage paired with hardware clock frequency.
    *   `cpu_utilization.png` - Average multi-core CPU usage profile.
    *   `memory_usage.png` - RAM and SWAP consumption.
    *   `temperature.png` - CPU and GPU thermal curves.
5.  **`scaling_comparison.csv`**: (Generated during `--auto-scale`) A summary table showing scaling numbers from 1 to 64 streams.

---

## Business Logic - UCB Alert Engine

Each stream feeds its detections to a dedicated instance of `UCBEngine`. The engine uses a simplified, high-performance Python port of the C++ logic:
1.  **Child & Adult Evaluation**: Determines `child_detected` (class 0 confidence >= 0.10) and `adult_detected` (class 1 confidence >= 0.10).
2.  **Unattended Status**: Flagged if `child_detected` is True and `adult_detected` is False.
3.  **Memory Reset**: On alert trigger (unattended count >= 26 frames over 85 seconds, max score >= 0.35, current score >= 0.30), the rolling window is immediately **wiped clean** to prevent stale detections from triggering duplicate alerts.
