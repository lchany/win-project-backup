import re
from pathlib import Path

GPU = Path(r"C:\project\win-project-backup\DongFeng\gpu去除随机性固定后loss.log")

N245 = {
    1: 435.7073, 2: 426.184, 3: 420.0385, 4: 421.9305, 5: 423.992,
    6: 414.2399, 7: 408.3089, 8: 395.3329, 9: 385.7871, 10: 381.2047,
    11: 347.9472, 12: 318.7897, 13: 306.8332, 14: 292.2801, 15: 281.4991,
    16: 276.8198, 17: 250.9507, 18: 261.1622, 19: 233.6606, 20: 232.7583,
    21: 223.5447, 22: 234.8167, 23: 219.232, 24: 199.9433, 25: 207.3541,
    26: 195.3782, 27: 189.4209, 28: 186.081, 29: 189.1717, 30: 182.4092,
}


def extract_gpu(path, max_it=30):
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        m = re.search(r"Iter \[(\d+)/", line)
        if not m:
            continue
        it = int(m.group(1))
        if it > max_it:
            continue
        lm = re.search(r"loss:\s*([0-9.eE+-]+)", line)
        if lm:
            out[it] = float(lm.group(1))
    return out


gpu = extract_gpu(GPU)
print("iter | npu_loss | gpu_loss | diff% | <=1%")
fail = []
for it in range(1, 31):
    g = gpu.get(it)
    n = N245.get(it)
    if g is None or n is None:
        continue
    d = (n - g) / g * 100
    ok = abs(d) <= 1.0
    if not ok:
        fail.append((it, d))
    mark = "OK" if ok else "FAIL"
    print(f"{it:4d} | {n:8.4f} | {g:8.4f} | {d:+6.2f}% | {mark}")

print(f"\nloss pass: {30 - len(fail)}/30")
print("fail iters:", fail)
