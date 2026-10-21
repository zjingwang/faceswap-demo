#!/usr/bin/env python3
"""OpenCV capture + face-swap in a background thread (no imshow).

Optionally mirrors the swapped feed to OBS Virtual Camera via pyvirtualcam,
so WeChat / Zoom can select "OBS Virtual Camera" as the webcam.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from ai_core import (
    ORT_PROVIDERS,
    detect_faces,
    ensure_models,
    extract_source_face,
    resize_keep_aspect,
    swap_faces,
)

VCAM_W, VCAM_H = 640, 480


def _frame_ok(frame) -> bool:
    if frame is None or frame.size == 0:
        return False
    return float(frame.mean()) > 8.0


def _open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    frame = None
    for _ in range(8):
        ok, frame = cap.read()
        if ok and _frame_ok(frame):
            return cap
    if frame is not None and frame.size:
        return cap
    cap.release()
    return None


def find_camera_index(preferred: int = -1) -> int:
    order = []
    if preferred >= 0:
        order.append(preferred)
    for i in range(6):
        if i not in order:
            order.append(i)
    for i in order:
        cap = _open_camera(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and _frame_ok(frame):
            return i
    raise RuntimeError(
        "找不到可用摄像头。请关闭其它占用摄像头的程序后再试。"
    )


class LiveCaptureSession:
    """OpenCV → AI → Gradio preview (+ optional OBS virtual camera)."""

    def __init__(self, width: int = 320, det_size: int = 160):
        self.width = max(128, int(width))
        self.det_size = int(det_size)
        self._lock = threading.Lock()
        self._preview: Optional[np.ndarray] = None
        self._status = "未启动"
        self._source = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cam_index = -1
        self._ai_ms: deque = deque(maxlen=30)
        self._swapped = 0
        self._frames = 0
        self._virtual_cam = False

    def set_source(self, src_rgb) -> str:
        if src_rgb is None:
            with self._lock:
                self._source = None
                self._status = "请先上传源脸。"
            return self._status
        try:
            face = extract_source_face(src_rgb, is_rgb=True)
            with self._lock:
                self._source = face
                dim = int(getattr(face, "embedding", np.array([])).shape[0] or 0)
                self._status = f"源脸已缓存 embedding({dim}-d)。"
            return self._status
        except Exception as e:
            with self._lock:
                self._source = None
                self._status = f"源脸提取失败: {e}"
            return self._status

    def get_preview(self):
        with self._lock:
            return None if self._preview is None else self._preview.copy()

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def is_running(self) -> bool:
        return self._running

    def start(self, camera: int = -1, virtual_cam: bool = False) -> str:
        if self._running:
            return "已在运行中。请先停止再改设置。"
        with self._lock:
            if self._source is None:
                return "请先上传源脸照片。"
        if virtual_cam:
            try:
                import pyvirtualcam  # noqa: F401
            except ImportError:
                return (
                    "未安装 pyvirtualcam。请执行: "
                    "pip install pyvirtualcam"
                )
        try:
            ensure_models(det_size=self.det_size, live=True)
            self._cam_index = find_camera_index(camera)
        except Exception as e:
            return f"启动失败: {e}"

        self._virtual_cam = bool(virtual_cam)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="live-cap", daemon=True)
        self._thread.start()
        extra = (
            " | 已推送到 OBS 虚拟摄像头 → 微信里选「OBS Virtual Camera」"
            if self._virtual_cam
            else ""
        )
        return (
            f"已启动本机摄像头 index={self._cam_index}，预览在本页。"
            f"{extra}"
        )

    def stop(self) -> str:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        with self._lock:
            self._status = "已停止。"
        return self._status

    def _open_vcam(self):
        import pyvirtualcam

        # Prefer OBS backend on Windows (requires OBS Studio installed once).
        last_err = None
        for backend in (None, "obs", "unitycapture"):
            try:
                kwargs = dict(
                    width=VCAM_W,
                    height=VCAM_H,
                    fps=20,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                )
                if backend:
                    kwargs["backend"] = backend
                cam = pyvirtualcam.Camera(**kwargs)
                print(
                    f"[live_capture] virtual cam OK backend={cam.backend} "
                    f"{cam.device} {cam.width}x{cam.height}",
                    flush=True,
                )
                return cam
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"无法打开虚拟摄像头（需已安装 OBS Studio）。详情: {last_err}"
        )

    def _loop(self):
        cap = _open_camera(self._cam_index)
        if cap is None:
            with self._lock:
                self._status = f"无法打开摄像头 {self._cam_index}"
                self._running = False
            return

        vcam = None
        vcam_err = None
        if self._virtual_cam:
            try:
                vcam = self._open_vcam()
            except Exception as e:
                vcam_err = str(e)
                print(f"[live_capture] virtual cam failed: {e}", flush=True)

        print(
            f"[live_capture] providers={ORT_PROVIDERS} cam={self._cam_index} "
            f"vcam={'on' if vcam else 'off'}",
            flush=True,
        )
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if not _frame_ok(frame):
                    with self._lock:
                        self._status = (
                            f"摄像头 {self._cam_index} 全黑，请停止后检查设备占用。"
                        )
                    time.sleep(0.05)
                    continue

                with self._lock:
                    source = self._source
                    width = self.width

                t0 = time.perf_counter()
                small = resize_keep_aspect(frame, width)
                display_bgr = small
                swapped = False
                if source is not None:
                    faces = detect_faces(small)
                    if faces:
                        target = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0])
                            * (f.bbox[3] - f.bbox[1]),
                        )
                        display_bgr = swap_faces(small, [target], source)
                        swapped = True
                ms = (time.perf_counter() - t0) * 1000.0

                # Clean RGB for virtual cam / WeChat (no HUD text).
                out_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                vcam_frame = cv2.resize(
                    out_rgb, (VCAM_W, VCAM_H), interpolation=cv2.INTER_LINEAR
                )
                if vcam is not None:
                    try:
                        vcam.send(vcam_frame)
                        vcam.sleep_until_next_frame()
                    except Exception as e:
                        vcam_err = str(e)

                # Gradio preview can show HUD.
                preview = vcam_frame.copy()
                cv2.putText(
                    preview,
                    f"{'SWAP' if swapped else 'CAM'} {ms:.0f}ms"
                    + (" | VCAM" if vcam else ""),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                with self._lock:
                    self._frames += 1
                    if swapped:
                        self._swapped += 1
                    self._ai_ms.append(ms)
                    avg = sum(self._ai_ms) / len(self._ai_ms)
                    fps = 1000.0 / avg if avg > 0 else 0.0
                    self._preview = preview
                    vmsg = " | 虚拟摄像头开" if vcam else ""
                    if vcam_err and vcam is None:
                        vmsg = f" | 虚拟摄像头失败: {vcam_err}"
                    self._status = (
                        f"本机 cam={self._cam_index}{vmsg} | "
                        f"AI {ms:.0f}ms (avg {avg:.0f}→{fps:.1f}FPS) | "
                        f"换脸 {self._swapped}/{self._frames} | 宽={width}"
                    )
        finally:
            cap.release()
            if vcam is not None:
                try:
                    vcam.close()
                except Exception:
                    pass
            self._running = False
