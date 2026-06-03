import os
import re
import sys
import time
import subprocess
import threading
import logging
import psutil
import random

logger = logging.getLogger("benchmark.monitor")

class SystemMonitor:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.running = False
        self.thread = None
        self.process = None
        self.lock = threading.Lock()
        
        # Latest telemetry snapshot
        self.metrics = {
            "timestamp": time.time(),
            "cpu_util": 0.0,
            "cpu_cores": [],
            "gpu_util": 0.0,
            "gpu_freq": 0,
            "gpu_mem_used": 0,
            "ram_used": 0.0, # GB
            "ram_free": 0.0, # GB
            "swap_used": 0.0, # GB
            "cpu_temp": 0.0,
            "gpu_temp": 0.0,
            "power_draw": 0.0, # W
            "disk_read_rate": 0.0, # MB/s
            "disk_write_rate": 0.0, # MB/s
            "nvpmodel": "UNKNOWN"
        }
        
        # Disk IO tracking
        self.last_disk_read = 0
        self.last_disk_write = 0
        self.last_disk_time = time.time()
        self._init_disk_io()
        
        # Ensure log folder exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.tegrastats_log_file = os.path.join(self.log_dir, "tegrastats.log")

    def _init_disk_io(self):
        try:
            io = psutil.disk_io_counters()
            if io:
                self.last_disk_read = io.read_bytes
                self.last_disk_write = io.write_bytes
        except Exception:
            pass
        self.last_disk_time = time.time()

    def _update_disk_rates(self):
        now = time.time()
        dt = now - self.last_disk_time
        if dt <= 0.01:
            return
            
        try:
            io = psutil.disk_io_counters()
            if io:
                read_diff = io.read_bytes - self.last_disk_read
                write_diff = io.write_bytes - self.last_disk_write
                
                # Convert bytes/s to MB/s
                self.metrics["disk_read_rate"] = max(0.0, (read_diff / dt) / (1024 * 1024))
                self.metrics["disk_write_rate"] = max(0.0, (write_diff / dt) / (1024 * 1024))
                
                self.last_disk_read = io.read_bytes
                self.last_disk_write = io.write_bytes
        except Exception:
            self.metrics["disk_read_rate"] = 0.0
            self.metrics["disk_write_rate"] = 0.0
            
        self.last_disk_time = now

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("System Monitor started.")

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("System Monitor stopped.")

    def get_latest_metrics(self) -> dict:
        with self.lock:
            # Refresh disk rate relative to caller time
            self._update_disk_rates()
            return self.metrics.copy()

    def _monitor_loop(self):
        # Determine if tegrastats is available
        import shutil
        if not shutil.which("tegrastats"):
            raise RuntimeError("tegrastats command-line utility not found in system path. Please ensure this is running on Jetson Orin NX.")

        logger.info("Running native tegrastats monitor...")
        self._run_tegrastats()

    def _run_tegrastats(self):
        try:
            # Write header to tegrastats.log
            with open(self.tegrastats_log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n--- Benchmark started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

            self.process = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            # Regex patterns for parsing tegrastats
            ram_pat = re.compile(r"RAM (\d+)/(\d+)MB")
            swap_pat = re.compile(r"SWAP (\d+)/(\d+)MB")
            cpu_pat = re.compile(r"CPU \[([^\]]+)\]")
            gpu_pat = re.compile(r"GR3D_FREQ\s+(\d+)%@?(\d+)?")
            gpu_alt_pat = re.compile(r"GR3D\s+(\d+)%")
            temp_cpu_pat = re.compile(r"CPU@([\d\.]+)C")
            temp_gpu_pat = re.compile(r"GPU@([\d\.]+)C")
            power_pat = re.compile(r"VDD_IN\s+(\d+)mW")
            
            while self.running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                # Write raw tegrastats to log
                with open(self.tegrastats_log_file, "a", encoding="utf-8") as lf:
                    lf.write(line)
                
                # Parse
                ram_m = ram_pat.search(line)
                swap_m = swap_pat.search(line)
                cpu_m = cpu_pat.search(line)
                gpu_m = gpu_pat.search(line) or gpu_alt_pat.search(line)
                t_cpu_m = temp_cpu_pat.search(line)
                t_gpu_m = temp_gpu_pat.search(line)
                pow_m = power_pat.search(line)
                
                with self.lock:
                    self.metrics["timestamp"] = time.time()
                    
                    if ram_m:
                        used, total = int(ram_m.group(1)), int(ram_m.group(2))
                        self.metrics["ram_used"] = used / 1024.0
                        self.metrics["ram_free"] = (total - used) / 1024.0
                    
                    if swap_m:
                        used = int(swap_m.group(1))
                        self.metrics["swap_used"] = used / 1024.0
                        
                    if cpu_m:
                        cores_str = cpu_m.group(1)
                        # cores_str looks like: "1%@1984,3%@1984,off,off,..."
                        cores = []
                        total_util = 0.0
                        active_count = 0
                        
                        for token in cores_str.split(','):
                            token = token.strip()
                            if token == "off" or not token:
                                cores.append(0.0)
                            else:
                                # Match % value
                                match = re.match(r"(\d+)%", token)
                                if match:
                                    util = float(match.group(1))
                                    cores.append(util)
                                    total_util += util
                                    active_count += 1
                                else:
                                    cores.append(0.0)
                                    
                        self.metrics["cpu_cores"] = cores
                        self.metrics["cpu_util"] = total_util / len(cores) if cores else 0.0
                        
                    if gpu_m:
                        self.metrics["gpu_util"] = float(gpu_m.group(1))
                        if len(gpu_m.groups()) > 1 and gpu_m.group(2):
                            self.metrics["gpu_freq"] = int(gpu_m.group(2))
                            
                    if t_cpu_m:
                        self.metrics["cpu_temp"] = float(t_cpu_m.group(1))
                    if t_gpu_m:
                        self.metrics["gpu_temp"] = float(t_gpu_m.group(1))
                        
                    if pow_m:
                        self.metrics["power_draw"] = float(pow_m.group(1)) / 1000.0 # to Watts
                    else:
                        # Parse other power rails if VDD_IN not found, e.g. POM_5V_IN
                        pom_m = re.search(r"POM_5V_IN\s+(\d+)/(\d+)", line)
                        if pom_m:
                            self.metrics["power_draw"] = float(pom_m.group(1)) / 1000.0
                            
            logger.info("tegrastats process terminated.")
        except Exception as e:
            logger.error(f"Error in tegrastats monitoring loop: {e}")
