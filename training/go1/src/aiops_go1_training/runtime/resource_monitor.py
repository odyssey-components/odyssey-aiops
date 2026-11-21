"""Sample the training process and its assigned GPU without blocking training."""

import ctypes
import json
import os
import threading
import time
from pathlib import Path
from typing import TypedDict


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class GPUProbe:
    def __init__(self):
        library = None
        for path in (
            "libnvidia-ml.so.1",
            "/usr/local/nvidia/lib64/libnvidia-ml.so.1",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        ):
            try:
                library = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if library is None:
            raise OSError("NVML library unavailable")
        self.lib = library
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
        ids = visible.split(",") if visible and visible not in ("all", "void", "none") else []
        if len(ids) == 1 and ids[0].startswith("GPU-"):
            result = self.lib.nvmlDeviceGetHandleByUUID(ids[0].encode(), ctypes.byref(self.handle))
        else:
            count = ctypes.c_uint()
            result = self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
            if result != 0 or count.value != 1:
                raise RuntimeError("cannot identify a single assigned GPU")
            result = self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
        if result != 0:
            raise RuntimeError("cannot access assigned GPU")

    def read(self):
        utilization, memory = _Utilization(), _Memory()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)) != 0:
            raise RuntimeError("GPU utilization unavailable")
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            raise RuntimeError("GPU memory unavailable")
        return utilization.gpu, memory.used / 1048576


def _rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("process RSS unavailable")


class _Metric(TypedDict):
    key: str
    label: str
    unit: str
    display: str
    value: float


class ResourceMonitor:
    def __init__(self, output: str | Path, interval: float = 5) -> None:
        self.output = Path(output) / "system-metrics.jsonl"
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="go1-resource-monitor", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        start = time.monotonic()
        previous_wall, previous_cpu = start, time.process_time()
        try:
            gpu = GPUProbe() if os.environ.get("AIOPS_GO1_DEVICE", "cuda") == "cuda" else None
        except (OSError, RuntimeError) as exc:
            gpu = None
            print(json.dumps({"event": "system.monitor.warning", "message": str(exc)}), flush=True)
        with self.output.open("a", buffering=1) as stream:
            while not self.stop_event.wait(self.interval):
                now, cpu = time.monotonic(), time.process_time()
                metrics: list[_Metric] = []

                def add(key: str, label: str, unit: str, value: float, *, metrics: list[_Metric] = metrics) -> None:
                    metrics.append(
                        {"key": key, "label": label, "unit": unit, "display": "line", "value": round(value, 3)}
                    )

                add(
                    "system.process_cpu_percent",
                    "训练进程 CPU 使用率",
                    "%",
                    100 * (cpu - previous_cpu) / max(now - previous_wall, 0.001),
                )
                previous_wall, previous_cpu = now, cpu
                try:
                    add("system.process_rss_mib", "训练进程内存 RSS", "MiB", _rss_mib())
                except (OSError, ValueError, RuntimeError) as exc:
                    print(json.dumps({"event": "system.monitor.warning", "message": str(exc)}), flush=True)
                if gpu:
                    try:
                        utilization, memory = gpu.read()
                        add("system.gpu_device_utilization_percent", "GPU 设备利用率", "%", utilization)
                        add("system.gpu_device_memory_mib", "GPU 设备显存占用", "MiB", memory)
                    except RuntimeError as exc:
                        print(json.dumps({"event": "system.monitor.warning", "message": str(exc)}), flush=True)
                event = {"event": "system.metrics", "attributes": {"step": round(now - start, 3), "metrics": metrics}}
                line = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                stream.write(line)
                os.write(1, line.encode())
