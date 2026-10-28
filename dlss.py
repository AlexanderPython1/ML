
"""
RTX 5060 GPU Benchmark
Синтетический тест CUDA, FP32/FP16 и обработки изображений.

Запуск:
    python rtx5060_benchmark.py

Результат:
    benchmark_results.csv

Важно:
    Это не настоящий DLSS benchmark.
"""

import csv
import gc
import platform
import statistics
import time
from datetime import datetime

import torch
import torch.nn.functional as F


# -------------------- Настройки --------------------

WARMUP = 5
REPEATS = 20
MATRIX_SIZE = 4096
IMAGE_SIZES = [(1920, 1080), (2560, 1440), (3840, 2160)]

results = []


def add_result(category, name, value, unit, extra=""):
    results.append({
        "category": category,
        "test": name,
        "value": round(float(value), 4),
        "unit": unit,
        "details": extra,
    })


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_cuda(name, fn, repeats=REPEATS):
    """Измерение времени CUDA-операции с прогревом."""
    for _ in range(WARMUP):
        output = fn()
    synchronize()

    times_ms = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        output = fn()
        synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    del output
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
    }


def report_timing(category, name, timing, details=""):
    add_result(category, name + " mean", timing["mean_ms"],
               "ms", details)
    add_result(category, name + " median", timing["median_ms"],
               "ms", details)
    add_result(category, name + " min", timing["min_ms"],
               "ms", details)


def main():
    print("=" * 60)
    print("GPU BENCHMARK")
    print("=" * 60)
    print("Date:", datetime.now().isoformat(timespec="seconds"))
    print("OS:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nCUDA недоступна.")
        print("Установи PyTorch с поддержкой CUDA и проверь драйвер.")
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print("GPU:", props.name)
    print("VRAM:", round(props.total_memory / 1024**3, 2), "GiB")
    print("CUDA runtime:", torch.version.cuda)

    add_result("system", "GPU name", 0, "text", props.name)
    add_result("system", "VRAM total",
               props.total_memory / 1024**3, "GiB")
    add_result("system", "PyTorch version", 0, "text", torch.__version__)
    add_result("system", "CUDA version", 0, "text",
               str(torch.version.cuda))

    # -------------------- 1. FP32 --------------------

    print("\n[1/4] FP32 matrix multiplication")
    n = MATRIX_SIZE
    a = torch.randn((n, n), device=device, dtype=torch.float32)
    b = torch.randn((n, n), device=device, dtype=torch.float32)

    timing = benchmark_cuda("fp32", lambda: torch.mm(a, b))
    report_timing("compute", "FP32 matmul", timing,
                  f"matrix={n}x{n}")

    # Теоретически выполнено 2*N^3 операций для одной матрицы.
    # Это оценка вычислительной скорости данного теста, не паспортный TFLOPS.
    tflops = (2 * n**3) / (timing["mean_ms"] / 1000) / 1e12
    add_result("compute", "FP32 effective throughput",
               tflops, "TFLOP/s", "2*N^3 / elapsed time")
    print(f"FP32: {timing['mean_ms']:.3f} ms, ~{tflops:.2f} TFLOP/s")

    del a, b
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------- 2. FP16 --------------------

    print("\n[2/4] FP16 matrix multiplication")
    a16 = torch.randn((n, n), device=device, dtype=torch.float16)
    b16 = torch.randn((n, n), device=device, dtype=torch.float16)

    timing = benchmark_cuda("fp16", lambda: torch.mm(a16, b16))
    report_timing("compute", "FP16 matmul", timing,
                  f"matrix={n}x{n}")

    tflops = (2 * n**3) / (timing["mean_ms"] / 1000) / 1e12
    add_result("compute", "FP16 effective throughput",
               tflops, "TFLOP/s", "not a direct Tensor Core measurement")
    print(f"FP16: {timing['mean_ms']:.3f} ms, ~{tflops:.2f} TFLOP/s")

    del a16, b16
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------- 3. Image processing --------------------

    print("\n[3/4] Image resize benchmark")
    print("Измеряется GPU-интерполяция, а не DLSS.")

    for width, height in IMAGE_SIZES:
        # Имитация низкого внутреннего разрешения:
        # приблизительно половина ширины и высоты.
        low_w = max(1, width // 2)
        low_h = max(1, height // 2)

        image = torch.rand(
            (1, 3, low_h, low_w), device=device, dtype=torch.float32
        )

        def upscale():
            return F.interpolate(
                image,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        timing = benchmark_cuda(
            f"resize_{width}x{height}", upscale, repeats=REPEATS
        )

        report_timing(
            "image_processing",
            f"Bilinear upscale {width}x{height}",
            timing,
            f"{low_w}x{low_h} -> {width}x{height}",
        )

        print(
            f"{width}x{height}: "
            f"{timing['mean_ms']:.3f} ms"
        )

        del image
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------- 4. Memory allocation / bandwidth proxy --------------------

    print("\n[4/4] GPU memory copy test")

    # Используется массив 256 MiB; если памяти не хватает,
    # уменьши размер до 64 или 128 MiB.
    num_elements = (256 * 1024**2) // 4
    src = torch.rand(num_elements, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    timing = benchmark_cuda(
        "memory_copy",
        lambda: dst.copy_(src),
        repeats=REPEATS,
    )
    report_timing("memory", "GPU device-to-device copy", timing,
                  "256 MiB payload")

    # Минимальная оценка: размер одного буфера / время.
    # Реальная передача читает src и пишет dst, то есть трафик выше.
    gb_per_sec = (src.numel() * src.element_size() / 1e9) / (
        timing["mean_ms"] / 1000
    )
    add_result("memory", "Copy payload throughput",
               gb_per_sec, "GB/s",
               "one-buffer payload; not VRAM peak bandwidth")

    print(f"Copy payload rate: {gb_per_sec:.2f} GB/s")

    del src, dst
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------- Save results --------------------

    with open("benchmark_results.csv", "w",
              newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "test", "value", "unit", "details"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved: benchmark_results.csv")
    print("GPU:", props.name)
    print("Peak allocated VRAM:",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1), "MiB")
    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()
