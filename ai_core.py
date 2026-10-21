#!/usr/bin/env python3
"""Shared DirectML face-swap engine (AMD Radeon / no CUDA).

Providers are fixed to DmlExecutionProvider then CPUExecutionProvider.
All insightface / inswapper inference goes through INFERENCE_LOCK.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
SWAPPER_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")

# AMD DirectML first; CPU fallback. Never CUDA.
ORT_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]

INFERENCE_LOCK = threading.Lock()

_app: Optional[FaceAnalysis] = None
_swapper = None
_sess_opts: Optional[ort.SessionOptions] = None


def build_sess_options(intra: int = 4, inter: int = 1) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, intra)
    opts.inter_op_num_threads = max(1, inter)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def ensure_models(
    det_size: int = 320,
    intra: int = 4,
    inter: int = 1,
    live: bool = False,
):
    """Lazy-load FaceAnalysis + inswapper once for the whole process.

    live=True skips genderage (unused by swap) for slightly lower webcam latency.
    """
    global _app, _swapper, _sess_opts
    if _app is not None and _swapper is not None:
        return _app, _swapper
    if not os.path.isfile(SWAPPER_PATH):
        raise RuntimeError(
            f"缺少模型 {SWAPPER_PATH}，请先运行: python download_models.py"
        )
    _sess_opts = build_sess_options(intra, inter)
    print(f"[ai_core] ORT providers = {ORT_PROVIDERS}", flush=True)
    print("[ai_core] 加载 buffalo_l ...", flush=True)
    fa_kwargs = dict(
        name="buffalo_l",
        root=BASE,
        providers=ORT_PROVIDERS,
        sess_options=_sess_opts,
    )
    if live:
        fa_kwargs["allowed_modules"] = ["detection", "recognition"]
    _app = FaceAnalysis(**fa_kwargs)
    _app.prepare(ctx_id=0, det_size=(det_size, det_size))
    print("[ai_core] 加载 inswapper_128.onnx ...", flush=True)
    _swapper = insightface.model_zoo.get_model(
        SWAPPER_PATH, providers=ORT_PROVIDERS, sess_options=_sess_opts
    )
    with INFERENCE_LOCK:
        _ = _app.get(np.zeros((240, 320, 3), dtype=np.uint8))
    print("[ai_core] 模型就绪 (DirectML)", flush=True)
    return _app, _swapper


def set_det_size(det_size: int) -> None:
    app, _ = ensure_models(det_size=det_size)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))


def extract_source_face(rgb_or_bgr, is_rgb: bool = True):
    """Detect faces and return the largest Face (embedding cached on the Face)."""
    app, _ = ensure_models()
    img = np.asarray(rgb_or_bgr)
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    with INFERENCE_LOCK:
        faces = app.get(img)
    if not faces:
        raise ValueError("未检测到人脸，请换更清晰、正面、光线充足的照片")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def detect_faces(bgr):
    app, _ = ensure_models()
    with INFERENCE_LOCK:
        return app.get(bgr)


def swap_faces(bgr, target_faces, source_face):
    """Swap source_face onto each face in target_faces; returns BGR image."""
    _, swapper = ensure_models()
    out = bgr
    with INFERENCE_LOCK:
        for face in target_faces:
            out = swapper.get(out, face, source_face, paste_back=True)
    return out


def resize_keep_aspect(bgr_or_rgb, width: int = 320):
    """Downscale so width == width (keep aspect). No-op if already smaller."""
    h, w = bgr_or_rgb.shape[:2]
    if w <= width:
        return bgr_or_rgb
    nh = max(1, int(h * width / w))
    return cv2.resize(bgr_or_rgb, (width, nh), interpolation=cv2.INTER_AREA)


def stretch_to(img, size_wh):
    """Stretch img back to (width, height)."""
    w, h = size_wh
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def sort_faces_left_to_right(faces):
    return sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) * 0.5)
