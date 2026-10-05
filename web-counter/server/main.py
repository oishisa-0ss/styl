import io
import math
import os
import sys
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from counter import IMGSZ, SIDE, ColonyDetector, Dish, count_photo, detect_dish

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
MODEL = "01_XMG_s"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?", allow_methods=["GET", "POST"])
det = ColonyDetector(threads=int(os.environ.get("ORT_THREADS", "0")) or None)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/count")
async def count(image: UploadFile = File(...), cx: float | None = Form(None), cy: float | None = Form(None),
                r: float | None = Form(None)):
    if any(v is not None and not math.isfinite(v) for v in (cx, cy, r)):
        raise HTTPException(400, "範囲の値が正しくありません")
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "写真が大きすぎます（15MB まで）")
    try:
        src = Image.open(io.BytesIO(data))
        if src.width * src.height > MAX_PIXELS:
            raise HTTPException(413, "写真の画素数が多すぎます")
        im = ImageOps.exif_transpose(src).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, "写真を読み込めませんでした")
    w, h = im.size
    t0 = time.perf_counter()
    found = True
    if cx is not None and cy is not None and r and r > 0:
        dish = Dish(min(max(cx, 0), w), min(max(cy, 0), h), min(max(r, min(w, h) * 0.05), max(w, h)))
    else:
        dish = detect_dish(im)
        if dish is None:
            found = False
            s = min(w, h)
            dish = Dish(w / 2, h / 2, s / 2 / 1.04)
    res = count_photo(im, det, dish=dish)
    l, t, s = res.crop
    k = s / SIDE
    boxes = [[round(l + x1 * k, 1), round(t + y1 * k, 1), round(l + x2 * k, 1), round(t + y2 * k, 1), round(sc, 3)]
             for x1, y1, x2, y2, sc in res.boxes]
    return {
        "w": w, "h": h, "found": found,
        "dish": [round(dish.cx, 1), round(dish.cy, 1), round(dish.r, 1)],
        "crop": [l, t, s], "boxes": boxes,
        "model": MODEL, "imgsz": IMGSZ, "ms": round((time.perf_counter() - t0) * 1000),
    }
