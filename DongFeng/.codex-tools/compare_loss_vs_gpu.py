import re
import json
from pathlib import Path

GPU = Path(r"C:\project\win-project-backup\DongFeng\gpu去除随机性固定后loss.log")

N244_JSON = r"""{"1": {"loss": 435.7073, "frame_0_loss_map_pts": 64.2216, "frame_0_loss_map_cls": 7.1506}, "2": {"loss": 426.184, "frame_0_loss_map_pts": 62.2385, "frame_0_loss_map_cls": 7.216}, "3": {"loss": 420.0385, "frame_0_loss_map_pts": 61.7379, "frame_0_loss_map_cls": 7.1524}, "4": {"loss": 421.9305, "frame_0_loss_map_pts": 62.4854, "frame_0_loss_map_cls": 7.1801}, "5": {"loss": 423.992, "frame_0_loss_map_pts": 61.5909, "frame_0_loss_map_cls": 7.1767}, "6": {"loss": 414.1729, "frame_0_loss_map_pts": 61.0158, "frame_0_loss_map_cls": 6.5008}, "7": {"loss": 408.2945, "frame_0_loss_map_pts": 60.5438, "frame_0_loss_map_cls": 5.2165}, "8": {"loss": 395.3416, "frame_0_loss_map_pts": 58.3632, "frame_0_loss_map_cls": 4.2733}, "9": {"loss": 385.8414, "frame_0_loss_map_pts": 57.0449, "frame_0_loss_map_cls": 3.657}, "10": {"loss": 381.0043, "frame_0_loss_map_pts": 55.7861, "frame_0_loss_map_cls": 2.7443}, "11": {"loss": 348.1484, "frame_0_loss_map_pts": 49.1698, "frame_0_loss_map_cls": 2.2384}, "12": {"loss": 319.086, "frame_0_loss_map_pts": 45.9768, "frame_0_loss_map_cls": 2.0879}, "13": {"loss": 306.7342, "frame_0_loss_map_pts": 41.3512, "frame_0_loss_map_cls": 1.6927}, "14": {"loss": 292.121, "frame_0_loss_map_pts": 38.2727, "frame_0_loss_map_cls": 1.6614}, "15": {"loss": 282.08, "frame_0_loss_map_pts": 37.9177, "frame_0_loss_map_cls": 1.3792}, "16": {"loss": 275.9988, "frame_0_loss_map_pts": 35.9493, "frame_0_loss_map_cls": 1.1136}, "17": {"loss": 250.0585, "frame_0_loss_map_pts": 33.4726, "frame_0_loss_map_cls": 0.9267}, "18": {"loss": 259.3156, "frame_0_loss_map_pts": 35.329, "frame_0_loss_map_cls": 0.9502}, "19": {"loss": 233.8145, "frame_0_loss_map_pts": 32.6637, "frame_0_loss_map_cls": 1.0511}, "20": {"loss": 232.7621, "frame_0_loss_map_pts": 32.0604, "frame_0_loss_map_cls": 0.9912}, "21": {"loss": 222.6945, "frame_0_loss_map_pts": 29.4079, "frame_0_loss_map_cls": 0.867}, "22": {"loss": 234.3214, "frame_0_loss_map_pts": 32.7916, "frame_0_loss_map_cls": 0.7423}, "23": {"loss": 219.2611, "frame_0_loss_map_pts": 30.6411, "frame_0_loss_map_cls": 0.8459}, "24": {"loss": 198.1511, "frame_0_loss_map_pts": 27.2397, "frame_0_loss_map_cls": 0.7627}, "25": {"loss": 205.0388, "frame_0_loss_map_pts": 28.4507, "frame_0_loss_map_cls": 0.745}, "26": {"loss": 194.7383, "frame_0_loss_map_pts": 27.2013, "frame_0_loss_map_cls": 0.6549}, "27": {"loss": 188.884, "frame_0_loss_map_pts": 26.3032, "frame_0_loss_map_cls": 0.6432}, "28": {"loss": 186.5164, "frame_0_loss_map_pts": 25.6718, "frame_0_loss_map_cls": 0.6517}, "29": {"loss": 187.7475, "frame_0_loss_map_pts": 26.9198, "frame_0_loss_map_cls": 0.6515}, "30": {"loss": 180.0368, "frame_0_loss_map_pts": 24.9917, "frame_0_loss_map_cls": 0.5435}}"""


def extract_gpu(path):
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        m = re.search(r"Iter \[(\d+)/", line)
        if not m:
            continue
        it = int(m.group(1))
        if it > 30:
            continue
        row = {}
        for k in ("loss", "frame_0_loss_map_pts", "frame_0_loss_map_cls"):
            lm = re.search(rf"{re.escape(k)}:\s*([0-9.eE+-]+)", line)
            if lm:
                row[k] = float(lm.group(1))
        if row:
            out[it] = row
    return out


def pct(n, g):
    return (n - g) / g * 100 if g else 0.0


gpu = extract_gpu(GPU)
n244 = {int(k): v for k, v in json.loads(N244_JSON).items()}

print("iter | metric | NPU | GPU | diff% | <=1%")
fail_loss = []
for it in range(1, 31):
    g = gpu.get(it)
    n = n244.get(it)
    if not g or not n:
        continue
    for key in ("loss", "frame_0_loss_map_pts", "frame_0_loss_map_cls"):
        d = pct(n[key], g[key])
        ok = abs(d) <= 1.0
        if key == "loss" and not ok:
            fail_loss.append((it, d))
        mark = "OK" if ok else "FAIL"
        if key == "loss" or not ok:
            print(f"{it:4d} | {key:24s} | {n[key]:8.4f} | {g[key]:8.4f} | {d:+6.2f}% | {mark}")

print("\nloss fail iters:", fail_loss)
print("loss pass count:", 30 - len(fail_loss), "/ 30")
