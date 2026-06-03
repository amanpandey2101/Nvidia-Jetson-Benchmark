import os
import sys
import time
import json
import csv
import argparse
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
import psutil

# Local imports
from dataset import CSVDataset
from ucb_engine import UCBEngine
from trt_engine import TensorRTEngine, preprocess_image, nms
from monitor import SystemMonitor

# Global variables for stop event and locks
stop_event = threading.Event()
print_lock = threading.Lock()

# Custom matplotlib styling for premium design aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '-'
})

# Color palette
COLORS = {
    'primary': '#6C5CE7',    # Vibrant Purple
    'secondary': '#00CEC9',  # Cyan
    'accent': '#FF7675',     # Coral
    'success': '#2ECC71',    # Green
    'dark': '#2D3436',       # Dark Charcoal
    'light': '#DFE6E9',      # Light Grey
}

def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler (Structured Debug Log)
    fh = logging.FileHandler(os.path.join(log_dir, "benchmark.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(threadName)s] [%(levelname)s] %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler (Clean Output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    
    return logger

def save_plots(telemetry_history: list, latency_samples: list, output_dir: str):
    """
    Generates premium, wowed-design charts from the collected benchmark metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data to plot
    if not telemetry_history:
        return
        
    t_times = [t["timestamp"] - telemetry_history[0]["timestamp"] for t in telemetry_history]
    
    # --- Plot 1: FPS & Throughput over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fps_vals = [t.get("fps", 0.0) for t in telemetry_history]
    ax.plot(t_times, fps_vals, color=COLORS['primary'], linewidth=2.5, label='FPS')
    ax.fill_between(t_times, fps_vals, color=COLORS['primary'], alpha=0.15)
    ax.set_title("Global FPS & Throughput Over Time", pad=15, weight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frames Per Second")
    ax.set_xlim(0, max(t_times) if max(t_times) > 0 else 1)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fps_over_time.png"), dpi=200)
    plt.close()
    
    # --- Plot 2: Latency over time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    if latency_samples:
        # Sort samples by time offset
        sorted_samples = sorted(latency_samples, key=lambda x: x["timestamp"])
        t0 = sorted_samples[0]["timestamp"]
        sample_times = [s["timestamp"] - t0 for s in sorted_samples]
        latencies = [s["latency"] * 1000.0 for s in sorted_samples] # to ms
        
        # Draw moving average of latency
        window = min(50, len(latencies))
        if window > 1:
            m_avg = np.convolve(latencies, np.ones(window)/window, mode='valid')
            ax.plot(sample_times[window-1:], m_avg, color=COLORS['accent'], linewidth=2.5, label=f'M-Avg Latency (W={window})')
        ax.scatter(sample_times, latencies, color=COLORS['primary'], alpha=0.15, s=8, label='Raw Detections')
        ax.set_title("Inference Latency Distribution Over Time", pad=15, weight='bold')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Latency (milliseconds)")
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, "No latency samples available", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_over_time.png"), dpi=200)
    plt.close()

    # --- Plot 3: GPU Utilization & Frequency ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    gpu_util = [t.get("gpu_util", 0.0) for t in telemetry_history]
    gpu_freq = [t.get("gpu_freq", 0) for t in telemetry_history]
    
    ax1.plot(t_times, gpu_util, color=COLORS['primary'], linewidth=2.5, label='GPU Util %')
    ax1.fill_between(t_times, gpu_util, color=COLORS['primary'], alpha=0.15)
    ax1.set_ylabel("GPU Utilization (%)", color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_ylim(0, 105)
    
    ax2 = ax1.twinx()
    ax2.plot(t_times, gpu_freq, color=COLORS['secondary'], linewidth=1.5, linestyle='--', label='GPU Freq (MHz)')
    ax2.set_ylabel("GPU Frequency (MHz)", color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
    
    plt.title("GPU Load & Frequency", pad=15, weight='bold')
    ax1.set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_utilization.png"), dpi=200)
    plt.close()

    # --- Plot 4: CPU Core and Average Utilization ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cpu_util = [t.get("cpu_util", 0.0) for t in telemetry_history]
    ax.plot(t_times, cpu_util, color=COLORS['dark'], linewidth=2.5, label='Avg CPU Load')
    ax.fill_between(t_times, cpu_util, color=COLORS['dark'], alpha=0.10)
    ax.set_title("CPU Utilization Profile", pad=15, weight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cpu_utilization.png"), dpi=200)
    plt.close()

    # --- Plot 5: RAM & SWAP Memory Usage ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ram_used = [t.get("ram_used", 0.0) for t in telemetry_history]
    swap_used = [t.get("swap_used", 0.0) for t in telemetry_history]
    
    ax.plot(t_times, ram_used, color=COLORS['primary'], linewidth=2.5, label='RAM Usage')
    ax.fill_between(t_times, ram_used, color=COLORS['primary'], alpha=0.15)
    if any(s > 0.05 for s in swap_used):
        ax.plot(t_times, swap_used, color=COLORS['accent'], linewidth=2.0, linestyle=':', label='Swap Usage')
        ax.fill_between(t_times, swap_used, color=COLORS['accent'], alpha=0.10)
        
    ax.set_title("Memory Demands Over Time", pad=15, weight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Memory Consumption (GB)")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage.png"), dpi=200)
    plt.close()

    # --- Plot 6: Thermals (CPU/GPU Temperatures) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cpu_temp = [t.get("cpu_temp", 0.0) for t in telemetry_history]
    gpu_temp = [t.get("gpu_temp", 0.0) for t in telemetry_history]
    
    ax.plot(t_times, cpu_temp, color=COLORS['accent'], linewidth=2.0, label='CPU Temp')
    ax.plot(t_times, gpu_temp, color=COLORS['primary'], linewidth=2.0, label='GPU Temp')
    ax.set_title("Thermal Zones Analysis", pad=15, weight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature.png"), dpi=200)
    plt.close()


def run_stream_worker(stream_id: int, 
                      engine, 
                      dataset_images: list, 
                      num_streams: int, 
                      args, 
                      results_list: list, 
                      ready_event: threading.Event):
    """
    Simulates a single camera feed stream running continuous inference.
    """
    logger = logging.getLogger(f"benchmark.stream-{stream_id}")
    
    # Thread-specific TensorRT execution context & UCB Engine
    ucb = None
    try:
        context = engine.create_execution_context()
        if not args.disable_ucb:
            ucb = UCBEngine(
                window_seconds=85,
                min_frames_in_window=args.min_frames,
                max_score_threshold=args.conf_threshold * 3.5, # Map standard conf to C++ equivalents
                last_score_threshold=args.conf_threshold * 3.0,
                alert_cooldown_seconds=60,
                ucb_required_fraction=0.75,
                child_confidence_threshold=args.conf_threshold,
                adult_confidence_threshold=args.conf_threshold
            )
    except Exception as e:
        logger.error(f"Failed to create context/engine for stream: {e}")
        ready_event.set()
        return

    # Signal ready
    ready_event.set()
    
    # Set reference to populated result dictionary
    res = results_list[stream_id]
    
    # Open-loop request timing
    target_rps = args.target_rps
    if target_rps:
        # Offsets start time to spread request spikes across streams
        next_inference_time = time.time() + (stream_id * (1.0 / (target_rps * num_streams)))
    else:
        next_inference_time = time.time()
        
    step = 0
    num_images = len(dataset_images)
    
    logger.info(f"Stream {stream_id} active.")
    
    while not stop_event.is_set():
        if args.images_per_stream and res["images_processed"] >= args.images_per_stream:
            break
            
        now = time.time()
        
        # Open-loop pacing: Wait until scheduled inference execution time
        if target_rps:
            delay = next_inference_time - now
            if delay > 0.001:
                time.sleep(delay)
                now = time.time()
            next_inference_time = max(now, next_inference_time) + (1.0 / target_rps)
            
        # Get round-robin preprocessed image or URL
        img_idx = (stream_id + step * num_streams) % num_images
        
        t_start = time.time()
        http_latency = 0.0
        
        if args.live_fetch or args.live_url:
            url = dataset_images[img_idx]
            raw_path = url
            try:
                import io
                import requests
                t_http_start = time.time()
                resp = requests.get(url, timeout=5)
                t_http_end = time.time()
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP Status {resp.status_code}")
                http_latency = t_http_end - t_http_start
                
                target_size = (engine.input_width, engine.input_height)
                preprocessed_img = preprocess_image(io.BytesIO(resp.content), target_size=target_size)
            except Exception as e:
                logger.error(f"Stream {stream_id} HTTP fetch/preprocess failed for {url}: {e}")
                time.sleep(0.1)
                step += 1
                continue
        else:
            preprocessed_img, raw_path = dataset_images[img_idx]
        
        # Execute Native Inference
        try:
            t_infer_start = time.time()
            detections = context.infer(preprocessed_img, args.conf_threshold, args.iou_threshold)
            t_infer_end = time.time()
            
            # Feed simplified UCB alerts logic
            alert = False
            if ucb:
                alert = ucb.feed(detections)
                if alert:
                    res["alerts_fired"] += 1
                    with print_lock:
                        logger.info(f"*** ALERT TRIGGERED on Stream {stream_id} ***")
            
            if args.save_detections:
                detection_log = {
                    "timestamp": t_start,
                    "stream_id": stream_id,
                    "image": raw_path,
                    "detections": [
                        {
                            "class_id": d["class_id"],
                            "confidence": round(d["confidence"], 4),
                            "box": [round(c, 2) for c in d["box"]]
                        }
                        for d in detections
                    ]
                }
                stream_log_path = os.path.join(args.output_dir, f"detections_stream_{stream_id}.jsonl")
                try:
                    with open(stream_log_path, "a", encoding="utf-8") as f_det:
                        f_det.write(json.dumps(detection_log) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write detection log: {e}")
                    
            t_end = time.time()
            
            res["latencies"].append({
                "timestamp": t_start,
                "latency": t_end - t_start,
                "infer_latency": t_infer_end - t_infer_start,
                "http_latency": http_latency,
                "alert": alert
            })
            res["images_processed"] += 1
            
        except Exception as e:
            logger.error(f"Inference crash in loop: {e}")
            time.sleep(0.1)
            
        step += 1

    logger.info(f"Stream {stream_id} completed. Processed: {res['images_processed']} images.")


def run_benchmark(engine, dataset_images: list, streams_count: int, args) -> dict:
    """
    Launches and coordinates stream workers. Returns compiled statistics.
    """
    logger = logging.getLogger("benchmark.run")
    logger.info(f"Starting benchmark: {streams_count} streams, target_rps={args.target_rps}, duration={args.duration}s")
    
    stop_event.clear()
    
    # Allocation structures
    workers = []
    results = [None] * streams_count
    for i in range(streams_count):
        results[i] = {
            "stream_id": i,
            "images_processed": 0,
            "alerts_fired": 0,
            "latencies": []
        }
        
    ready_events = [threading.Event() for _ in range(streams_count)]
    
    # Launch background system monitor
    monitor = SystemMonitor(log_dir="logs")
    monitor.start()
    
    # Start thread pool
    for i in range(streams_count):
        t = threading.Thread(
            target=run_stream_worker,
            args=(i, engine, dataset_images, streams_count, args, results, ready_events[i]),
            name=f"Stream-{i}",
            daemon=True
        )
        workers.append(t)
        t.start()
        
    # Wait for all stream contexts to set up
    for ev in ready_events:
        ev.wait()
        
    t_start = time.time()
    telemetry_history = []
    
    # Telemetry logging thread
    def telemetry_loop():
        # Delay slightly to avoid catching initial context overhead
        time.sleep(0.5)
        # Warmup loop: calculate processed count delta
        last_total_processed = 0
        while not stop_event.is_set():
            t_tick = time.time()
            metrics = monitor.get_latest_metrics()
            
            # Calculate current throughput FPS
            curr_processed = 0
            for r in results:
                if r:
                    curr_processed += r["images_processed"]
            
            processed_delta = curr_processed - last_total_processed
            last_total_processed = curr_processed
            
            # Record current telemetry
            metrics["fps"] = float(processed_delta) # Delta per 1-sec tick
            telemetry_history.append(metrics)
            time.sleep(1.0)
            
    tel_thread = threading.Thread(target=telemetry_loop, daemon=True)
    tel_thread.start()
    
    # Main execution timer
    if args.duration:
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt caught in main loop. Shutting down gracefully...")
    else:
        # Loop forever or wait for images_per_stream criteria
        try:
            while not stop_event.is_set():
                # Check if all workers completed their workload
                all_done = True
                for t in workers:
                    if t.is_alive():
                        all_done = False
                        break
                if all_done:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt caught in main loop. Shutting down...")
            
    # Trigger stop signal
    stop_event.set()
    
    # Join threads
    for t in workers:
        t.join(timeout=3)
    monitor.stop()
    
    t_end = time.time()
    total_time = t_end - t_start
    
    # Aggregate and compile report data
    global_latencies = []
    global_infer_latencies = []
    total_images = 0
    total_alerts = 0
    stream_summaries = []
    
    for r in results:
        if r is None:
            continue
        s_id = r["stream_id"]
        imgs = r["images_processed"]
        alrt = r["alerts_fired"]
        s_lats = [s["latency"] for s in r["latencies"]]
        s_inf = [s["infer_latency"] for s in r["latencies"]]
        s_http = [s.get("http_latency", 0.0) for s in r["latencies"]]
        
        total_images += imgs
        total_alerts += alrt
        global_latencies.extend(r["latencies"])
        global_infer_latencies.extend(s_inf)
        
        s_lat_ms = [l * 1000.0 for l in s_lats]
        s_http_ms = [l * 1000.0 for l in s_http]
        stream_summaries.append({
            "stream_id": s_id,
            "images_processed": imgs,
            "alerts_fired": alrt,
            "fps": imgs / total_time,
            "latency_avg_ms": np.mean(s_lat_ms) if s_lat_ms else 0.0,
            "latency_p95_ms": np.percentile(s_lat_ms, 95) if s_lat_ms else 0.0,
            "latency_max_ms": np.max(s_lat_ms) if s_lat_ms else 0.0,
            "http_avg_ms": np.mean(s_http_ms) if s_http_ms else 0.0
        })

    raw_latencies_ms = [item["latency"] * 1000.0 for item in global_latencies]
    raw_infer_ms = [item * 1000.0 for item in global_infer_latencies]
    raw_http_ms = [item.get("http_latency", 0.0) * 1000.0 for item in global_latencies]
    
    # Global aggregates
    global_fps = total_images / total_time
    avg_latency = np.mean(raw_latencies_ms) if raw_latencies_ms else 0.0
    p50_latency = np.percentile(raw_latencies_ms, 50) if raw_latencies_ms else 0.0
    p95_latency = np.percentile(raw_latencies_ms, 95) if raw_latencies_ms else 0.0
    p99_latency = np.percentile(raw_latencies_ms, 99) if raw_latencies_ms else 0.0
    min_latency = np.min(raw_latencies_ms) if raw_latencies_ms else 0.0
    max_latency = np.max(raw_latencies_ms) if raw_latencies_ms else 0.0
    
    avg_infer = np.mean(raw_infer_ms) if raw_infer_ms else 0.0
    p95_infer = np.percentile(raw_infer_ms, 95) if raw_infer_ms else 0.0
    
    avg_http = np.mean(raw_http_ms) if raw_http_ms else 0.0
    p95_http = np.percentile(raw_http_ms, 95) if raw_http_ms else 0.0
    
    # Hardware metrics aggregates (skipping initial 1-sec noise)
    valid_hardware = telemetry_history[1:] if len(telemetry_history) > 1 else telemetry_history
    avg_gpu_util = np.mean([h.get("gpu_util", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    avg_cpu_util = np.mean([h.get("cpu_util", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    avg_ram_used = np.mean([h.get("ram_used", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    max_ram_used = np.max([h.get("ram_used", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    avg_cpu_temp = np.mean([h.get("cpu_temp", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    avg_gpu_temp = np.mean([h.get("gpu_temp", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    max_power = np.max([h.get("power_draw", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    avg_power = np.mean([h.get("power_draw", 0.0) for h in valid_hardware]) if valid_hardware else 0.0
    
    report = {
        "streams": streams_count,
        "duration_seconds": total_time,
        "total_images_processed": total_images,
        "total_alerts_fired": total_alerts,
        "global_fps": global_fps,
        "latency_avg_ms": avg_latency,
        "latency_p50_ms": p50_latency,
        "latency_p95_ms": p95_latency,
        "latency_p99_ms": p99_latency,
        "latency_min_ms": min_latency,
        "latency_max_ms": max_latency,
        "infer_avg_ms": avg_infer,
        "infer_p95_ms": p95_infer,
        "http_avg_ms": avg_http,
        "http_p95_ms": p95_http,
        "gpu_util_avg": avg_gpu_util,
        "cpu_util_avg": avg_cpu_util,
        "ram_used_avg": avg_ram_used,
        "ram_used_max": max_ram_used,
        "cpu_temp_avg": avg_cpu_temp,
        "gpu_temp_avg": avg_gpu_temp,
        "power_draw_avg": avg_power,
        "power_draw_max": max_power,
        "stream_details": stream_summaries,
        "telemetry_history": telemetry_history,
        "latency_samples": global_latencies
    }
    
    return report

def write_reports(report: dict, args):
    """
    Saves CSV, JSON, and charts to reports/ folder.
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save JSON Report
    json_path = os.path.join(out_dir, "benchmark_report.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, default=lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
    # 2. Save CSV Report
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "streams", "fps", "latency_avg", "latency_p95", 
            "latency_p99", "gpu_util", "cpu_util", "ram_used", "gpu_temp", "cpu_temp"
        ])
        for t in report["telemetry_history"]:
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t["timestamp"])),
                report["streams"],
                f"{t.get('fps', 0.0):.2f}",
                f"{report['latency_avg_ms']:.2f}",
                f"{report['latency_p95_ms']:.2f}",
                f"{report['latency_p99_ms']:.2f}",
                f"{t.get('gpu_util', 0.0):.1f}",
                f"{t.get('cpu_util', 0.0):.1f}",
                f"{t.get('ram_used', 0.0):.2f}",
                f"{t.get('gpu_temp', 0.0):.1f}",
                f"{t.get('cpu_temp', 0.0):.1f}"
            ])
            
    # 3. Generate Charts
    save_plots(report["telemetry_history"], report["latency_samples"], out_dir)
    
    logger = logging.getLogger("benchmark.reporter")
    logger.info(f"Saved JSON report: {json_path}")
    logger.info(f"Saved CSV results: {csv_path}")
    logger.info(f"Saved performance charts under: {out_dir}/")


def print_console_summary(report: dict):
    """
    Renders the beautiful benchmark console output.
    """
    print("\n" + "="*50)
    print(f" BENCHMARK RUN SUMMARY (Streams: {report['streams']})")
    print("="*50)
    print(f"Total Images Processed: {report['total_images_processed']}")
    print(f"Duration:               {report['duration_seconds']:.2f} seconds")
    print(f"Total Alerts Triggered: {report['total_alerts_fired']}")
    print(f"Global Throughput (FPS):{report['global_fps']:.2f}")
    print("-"*50)
    print(f"Inference Latencies (Total Loop):")
    print(f"  Average:              {report['latency_avg_ms']:.2f} ms")
    print(f"  P95 Percentile:       {report['latency_p95_ms']:.2f} ms")
    print(f"  P99 Percentile:       {report['latency_p99_ms']:.2f} ms")
    print(f"  Min / Max:            {report['latency_min_ms']:.2f} / {report['latency_max_ms']:.2f} ms")
    print(f"Native Model Exec (TRT context only):")
    print(f"  Average:              {report['infer_avg_ms']:.2f} ms")
    print(f"  P95 Percentile:       {report['infer_p95_ms']:.2f} ms")
    if report.get('http_avg_ms', 0.0) > 0.0:
        print(f"HTTP Snapshot Fetch Latency:")
        print(f"  Average:              {report['http_avg_ms']:.2f} ms")
        print(f"  P95 Percentile:       {report['http_p95_ms']:.2f} ms")
    print("-"*50)
    print("Average Hardware Telemetry:")
    print(f"  GPU Utilization:      {report['gpu_util_avg']:.1f}%")
    print(f"  CPU Utilization:      {report['cpu_util_avg']:.1f}%")
    print(f"  RAM Usage (Max):      {report['ram_used_avg']:.2f} GB ({report['ram_used_max']:.2f} GB)")
    print(f"  GPU Temp:             {report['gpu_temp_avg']:.1f} °C")
    print(f"  CPU Temp:             {report['cpu_temp_avg']:.1f} °C")
    print(f"  Average Power Draw:   {report['power_draw_avg']:.2f} W")
    print("="*50 + "\n")


def run_scaling_benchmark(engine, dataset_images: list, args):
    """
    Auto-scales streams (1, 2, 4, 8, 16, 32, 64) until hardware limiters hit.
    """
    logger = logging.getLogger("benchmark.scaler")
    logger.info("Initializing Automatic Stream Scaling Benchmark Mode...")
    
    stream_counts = [1, 2, 4, 8, 16, 32, 64, 100, 128]
    scaling_results = []
    
    # Run duration for each scaling step
    args.duration = 30 # standard 30s sample
    
    for streams in stream_counts:
        logger.info(f"\nScaling Step: Running with {streams} streams...")
        
        report = run_benchmark(engine, dataset_images, streams, args)
        
        step_res = {
            "streams": streams,
            "fps": report["global_fps"],
            "latency_avg": report["latency_avg_ms"],
            "latency_p95": report["latency_p95_ms"],
            "gpu_util": report["gpu_util_avg"],
            "ram_used": report["ram_used_max"],
            "gpu_temp": report["gpu_temp_avg"]
        }
        scaling_results.append(step_res)
        
        print_console_summary(report)
        
        # Check bottleneck cutoffs
        gpu_cutoff = step_res["gpu_util"] >= 95.0
        ram_cutoff = step_res["ram_used"] >= (psutil.virtual_memory().total * 0.9 / (1024*1024*1024))
        latency_cutoff = step_res["latency_p95"] > 150.0
        
        cutoff_triggered = []
        if gpu_cutoff:
            cutoff_triggered.append("GPU Utilization (>= 95%)")
        if ram_cutoff:
            cutoff_triggered.append("RAM Utilization (>= 90%)")
        if latency_cutoff:
            cutoff_triggered.append("P95 Latency (> 150ms)")
            
        if cutoff_triggered:
            logger.warning(f"Bottleneck cutoff reached at {streams} streams: {', '.join(cutoff_triggered)}")
            break
            
        # Add a short cooling pause between runs
        logger.info("Cooldown pause (5s) for thermal recovery...")
        time.sleep(5)
        
    # Generate Comparison Table
    print("\n" + "="*80)
    print(" STREAM SCALING MODE COMPARISON TABLE")
    print("="*80)
    print(f"{'Streams':<8} | {'Throughput (FPS)':<18} | {'Avg Latency':<12} | {'P95 Latency':<12} | {'GPU %':<8} | {'Max RAM':<10}")
    print("-"*80)
    for res in scaling_results:
        print(
            f"{res['streams']:<8} | "
            f"{res['fps']:<18.2f} | "
            f"{res['latency_avg']:<10.2f} ms | "
            f"{res['latency_p95']:<10.2f} ms | "
            f"{res['gpu_util']:<6.1f}% | "
            f"{res['ram_used']:<8.2f} GB"
        )
    print("="*80 + "\n")
    
    # Save scaling results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    scaling_csv = os.path.join(args.output_dir, "scaling_comparison.csv")
    with open(scaling_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["streams", "fps", "latency_avg", "latency_p95", "gpu_util", "ram_used", "gpu_temp"])
        for res in scaling_results:
            writer.writerow([
                res["streams"], f"{res['fps']:.2f}", f"{res['latency_avg']:.2f}",
                f"{res['latency_p95']:.2f}", f"{res['gpu_util']:.1f}", f"{res['ram_used']:.2f}", f"{res['gpu_temp']:.1f}"
            ])
    logger.info(f"Saved auto-scaling comparison: {scaling_csv}")


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin NX TensorRT & UCB Alerts Benchmarking Tool")
    
    # Core Options
    parser.add_argument("--model", type=str, default="model.engine", help="Path to prebuilt TensorRT engine")
    parser.add_argument("--model-type", type=str, default="yolov5s", choices=["yolov5s", "rtdetr"], help="Inference model post-processor type")
    parser.add_argument("--csv-path", type=str, default="UCB-DATA.csv", help="CSV data source containing image URLs")
    parser.add_argument("--image-dir", type=str, default="images", help="Local directory to cache downloaded images")
    parser.add_argument("--streams", type=int, default=1, help="Number of simulated camera streams to run in parallel")
    
    # Benchmark Duration Options (Mutually Exclusive in Logic)
    parser.add_argument("--duration", type=int, default=None, help="Benchmark run time in seconds")
    parser.add_argument("--images-per-stream", type=int, default=None, help="Fixed workload: process N images per stream then finish")
    
    # Pacing / Capacity Options
    parser.add_argument("--target-rps", type=float, default=None, help="Open-loop request rate limit per stream (FPS)")
    parser.add_argument("--auto-scale", action="store_true", help="Launch scaling benchmark across multiple stream loads")
    
    # Advanced Options
    parser.add_argument("--max-download", type=int, default=None, help="Maximum number of images to download/target from CSV source (defaults to all URLs in CSV)")
    parser.add_argument("--disable-ucb", action="store_true", help="Disable UCB alert business logic (run raw inference only)")
    parser.add_argument("--save-detections", action="store_true", help="Save detailed bounding boxes and confidence scores to disk frame-by-frame")
    parser.add_argument("--preload-limit", type=int, default=250, help="Maximum number of preprocessed images to load in RAM")
    parser.add_argument("--live-fetch", action="store_true", help="Fetch images dynamically over HTTP for each frame request instead of preloading in RAM")
    parser.add_argument("--live-url", type=str, default=None, help="A single camera HTTP snapshot URL to query dynamically for each frame")
    
    # Thresholds
    parser.add_argument("--conf-threshold", type=float, default=0.10, help="Model detection confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IOU overlap threshold (YOLO only)")
    parser.add_argument("--min-frames", type=int, default=35, help="Minimum frames in window required to trigger alert")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory where CSV, JSON, and PNG charts will be written")
    
    args = parser.parse_args()

    # Setup directories and logging
    setup_logger("logs")
    logger = logging.getLogger("benchmark.main")
    logger.info("Initializing Benchmark Utility...")

    # Load dataset
    if args.live_url:
        local_images = []
        max_download = 0
    else:
        dataset = CSVDataset(args.csv_path, args.image_dir)
        max_download = args.max_download if args.max_download is not None else len(dataset.urls)
        if args.live_fetch:
            local_images = []
            if not dataset.urls:
                logger.critical("No URLs found in CSV for live-fetch. Aborting.")
                sys.exit(1)
        else:
            local_images = dataset.download_and_cache(max_download=max_download)
            if not local_images:
                logger.critical("No local images available. Aborting.")
                sys.exit(1)

    # Initialize Engine
    try:
        engine = TensorRTEngine(args.model, args.model_type)
    except Exception as e:
        logger.critical(f"Failed to load TensorRT engine: {e}")
        sys.exit(1)

    # Preprocess & preload images in-memory to prevent disk I/O bottlenecks during live benchmark
    preloaded_images = []
    
    if args.live_url:
        preloaded_images = [args.live_url]
        logger.info(f"Running in live-fetch mode targeting single URL: {args.live_url}")
    elif args.live_fetch:
        preloaded_images = dataset.urls[:max_download]
        logger.info(f"Running in live-fetch mode targeting {len(preloaded_images)} URLs from CSV.")
    else:
        logger.info(f"Preloading and pre-processing up to {args.preload_limit} images in-memory to maximize GPU/TensorRT throughput...")
        preload_limit = min(args.preload_limit, len(local_images))
        
        # Use target input shape dimensions dynamically from the loaded engine
        target_size = (engine.input_width, engine.input_height)
        logger.info(f"Target preprocessing size determined from engine: {target_size}")
        
        for path in local_images[:preload_limit]:
            try:
                arr = preprocess_image(path, target_size=target_size)
                preloaded_images.append((arr, path))
            except Exception as e:
                logger.warning(f"Failed preprocessing {path}: {e}")
                
        if not preloaded_images:
            logger.critical("Failed to preload any valid benchmark images. Aborting.")
            sys.exit(1)
            
        logger.info(f"Loaded {len(preloaded_images)} preprocessed images into RAM. Ready for inference loop.")

    # Standardize run configurations
    if not args.duration and not args.images_per_stream:
        args.duration = 60 # Default to 60s benchmark
        logger.info("No duration or fixed image counts set. Defaulting to 60-second test.")

    # Run selection
    if args.auto_scale:
        run_scaling_benchmark(engine, preloaded_images, args)
    else:
        report = run_benchmark(engine, preloaded_images, args.streams, args)
        print_console_summary(report)
        write_reports(report, args)
        
    logger.info("Benchmark process complete.")


if __name__ == "__main__":
    main()
