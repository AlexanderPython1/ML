import time
import statistics
import torch
import torch.nn.functional as F

device = "cuda"

# Синтетический кадр 960x540
image = torch.rand(1, 3, 540, 960, device=device)

def upscale():
    return F.interpolate(
        image,
        size=(1080, 1920),
        mode="bilinear",
        align_corners=False
    )

for _ in range(10):
    result = upscale()

torch.cuda.synchronize()

times = []
for _ in range(30):
    start = time.perf_counter()
    result = upscale()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print("Среднее время:", round(statistics.mean(times), 3), "мс")
print("Медиана:", round(statistics.median(times), 3), "мс")
print("Эквивалентная скорость:",
      round(1000 / statistics.mean(times), 2), "FPS")