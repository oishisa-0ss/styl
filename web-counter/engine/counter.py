from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE = 1800
IMGSZ = 1280
MODEL_PATH = os.path.join(HERE, "models", f"01_XMG_s_{IMGSZ}.onnx")
CONF = 0.40
CONF_MIN = 0.05
IOU = 0.45
MAX_DET = 1000
INSIDE = 1.06


@dataclass
class Dish:
    cx: float
    cy: float
    r: float


@dataclass
class Result:
    size: tuple[int, int]
    dish: Dish | None
    crop: tuple[int, int, int]
    dish_in_crop: Dish | None
    boxes: list[list[float]] = field(default_factory=list)
    outside: list[list[float]] = field(default_factory=list)
    square: Image.Image | None = None

    def count(self, conf: float = CONF) -> int:
        return sum(1 for b in self.boxes if b[4] >= conf)


def load_photo(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def detect_dish(im: Image.Image) -> Dish | None:
    w, h = im.size
    k = 800 / max(w, h)
    g = cv2.cvtColor(np.asarray(im.resize((round(w * k), round(h * k)))), cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 5)
    short = min(g.shape)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=short, param1=80, param2=40,
                          minRadius=int(short * 0.12), maxRadius=int(short * 0.52))
    if cs is None:
        return None
    x, y, r = (float(v) / k for v in cs[0][0])
    return Dish(x, y, r)


def square_around(dish: Dish | None, w: int, h: int) -> tuple[int, int, int]:
    if dish is None:
        s = min(w, h)
        return (w - s) // 2, (h - s) // 2, s
    s = int(min(2 * dish.r * 1.04, w, h))
    l = int(min(max(dish.cx - s / 2, 0), w - s))
    t = int(min(max(dish.cy - s / 2, 0), h - s))
    return l, t, s


class ColonyDetector:
    def __init__(self, model_path: str = MODEL_PATH, threads: int | None = None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, square: Image.Image, conf: float = CONF_MIN) -> list[list[float]]:
        a = np.asarray(square)[:, :, ::-1]
        r = cv2.resize(a, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(r[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        cx, cy, w, h, s = self.sess.run(None, {self.input: x})[0][0]
        k = s > conf
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)[k]
        s = s[k]
        keep = _nms(b, s, IOU)[:MAX_DET]
        scale = square.width / IMGSZ
        return [[*(float(v) * scale for v in bb), float(sc)] for bb, sc in zip(b[keep], s[keep])]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=int)


def count_photo(im: Image.Image, detector: ColonyDetector, limit: int | None = None,
                dish: Dish | None = None) -> Result:
    if limit and max(im.size) > limit:
        k = limit / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.Resampling.LANCZOS)
        if dish:
            dish = Dish(dish.cx * k, dish.cy * k, dish.r * k)
    w, h = im.size
    if dish is None:
        dish = detect_dish(im)
    l, t, s = square_around(dish, w, h)
    square = im.crop((l, t, l + s, t + s)).resize((SIDE, SIDE), Image.Resampling.LANCZOS)
    k = SIDE / s
    dic = Dish((dish.cx - l) * k, (dish.cy - t) * k, dish.r * k) if dish else None
    res = Result(size=(w, h), dish=dish, crop=(l, t, s), dish_in_crop=dic, square=square)
    for b in detector(square):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if dic is None or (cx - dic.cx) ** 2 + (cy - dic.cy) ** 2 <= (dic.r * INSIDE) ** 2:
            res.boxes.append(b)
        else:
            res.outside.append(b)
    return res
