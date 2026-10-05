import os
import shutil
import sys
import tempfile

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "models", "01_XMG_s.pt")
IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
DST = os.path.join(HERE, "models", f"01_XMG_s_{IMGSZ}.onnx")

with tempfile.TemporaryDirectory() as tmp:
    pt = os.path.join(tmp, "01_XMG_s.pt")
    shutil.copy(SRC, pt)
    out = YOLO(pt).export(format="onnx", imgsz=IMGSZ, simplify=False, dynamic=False, half=False)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    shutil.move(out, DST)
print(DST, os.path.getsize(DST) // 1024, "KB")
