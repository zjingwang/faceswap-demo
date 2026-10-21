#!/usr/bin/env python3
"""Gradio webcam face-swap — synchronous drop-frame (no worker thread).

Gradio only advances the preview when the stream callback returns. A background
worker + always_last made the UI look frozen until you hit Stop.
"""
from __future__ import annotations

import threading
import time
from collections import deque

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


class WebcamSwap:
    def __init__(
        self,
        frame_width: int = 320,
        det_size: int = 160,
        det_interval: int = 1,
    ):
        self.width = max(128, int(frame_width))
        self.det_size = int(det_size)
        self.interval = max(1, int(det_interval))
        self._infer_lock = threading.Lock()
        self._preview = None
        self._status = "模型加载中…"
        self._source = None
        self._seq = 0
        self._processed = 0
        self._swapped = 0
        self._dropped = 0
        self._ai_ms: deque = deque(maxlen=30)

    def start(self):
        ensure_models(det_size=self.det_size, live=True)
        self._status = f"模型就绪 ({ORT_PROVIDERS})。上传源脸后开启摄像头。"

    def stop(self):
        pass

    def set_source(self, src_rgb):
        if src_rgb is None:
            self._source = None
            self._status = "请先上传源脸照片。"
            return self._status
        try:
            face = extract_source_face(src_rgb, is_rgb=True)
            self._source = face
            dim = int(getattr(face, "embedding", np.array([])).shape[0] or 0)
            self._status = f"源脸已缓存 embedding({dim}-d)。"
            return self._status
        except Exception as e:
            self._source = None
            self._status = f"源脸提取失败: {e}"
            return self._status

    def get_status(self) -> str:
        return self._status

    def predict_image_only(self, frame_rgb):
        """Called by Gradio every stream_every seconds — must return quickly."""
        if frame_rgb is None:
            return self._preview

        echo = resize_keep_aspect(np.asarray(frame_rgb), self.width)
        self._seq += 1

        # No source yet: still show live camera so UI never freezes on placeholder.
        if self._source is None:
            self._preview = echo
            self._status = "已收到摄像头画面。请上传源脸。"
            return echo

        # Drop frame if previous AI still running (never block Gradio).
        if not self._infer_lock.acquire(blocking=False):
            self._dropped += 1
            return self._preview if self._preview is not None else echo

        try:
            if self._seq % self.interval != 0 and self._preview is not None:
                return self._preview

            t0 = time.perf_counter()
            small_bgr = cv2.cvtColor(echo, cv2.COLOR_RGB2BGR)
            faces = detect_faces(small_bgr)
            if faces:
                target = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                out_bgr = swap_faces(small_bgr, [target], self._source)
                display = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                swapped = True
            else:
                display = echo
                swapped = False
            ms = (time.perf_counter() - t0) * 1000.0
            cv2.putText(
                display,
                f"{'SWAP' if swapped else 'NO FACE'} {ms:.0f}ms",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self._processed += 1
            if swapped:
                self._swapped += 1
            self._ai_ms.append(ms)
            avg = sum(self._ai_ms) / len(self._ai_ms)
            fps = 1000.0 / avg if avg > 0 else 0.0
            self._preview = display
            self._status = (
                f"AI {ms:.0f}ms (avg {avg:.0f} → {fps:.1f}FPS) | "
                f"drop {self._dropped} | swap {self._swapped}/{self._processed}"
            )
            return display
        except Exception as e:
            self._status = f"处理异常: {e}"
            return echo
        finally:
            self._infer_lock.release()
