#!/usr/bin/env python3
"""Webcam live face-swap session (DirectML): drop-frame + cached embedding."""
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
    """Latest-frame slot + single worker; drop frames while AI is busy.

    Gradio callbacks never wait on inference: busy → return last result,
    overwrite the single pending slot (no queue / backlog).
    """

    def __init__(
        self,
        frame_width: int = 256,
        det_size: int = 192,
        det_interval: int = 3,
    ):
        self.width = max(128, int(frame_width))
        self.det_size = int(det_size)
        self.interval = max(1, int(det_interval))
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest = None
        self._busy = False
        self._result = self._placeholder("WAIT FIRST FRAME")
        self._status = "模型加载中…"
        self._received = 0
        self._processed = 0
        self._swapped = 0
        self._dropped = 0
        self._source = None
        self._target = None
        self._seq = 0
        self._last_det_seq = -1
        self._ai_ms: deque = deque(maxlen=20)
        self._cb_times: deque = deque(maxlen=40)
        self._last_cb_t = 0.0
        self._stop = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _placeholder(text: str, w: int = 320, h: int = 240) -> np.ndarray:
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        cv2.putText(
            img, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (40, 40, 40), 1, cv2.LINE_AA,
        )
        return img

    def start(self):
        ensure_models(det_size=self.det_size)
        with self._lock:
            self._status = (
                f"模型就绪 (providers={ORT_PROVIDERS})。请上传源脸并开启摄像头。"
            )
        self._thread = threading.Thread(
            target=self._workloop, name="webcam-swap", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=1.5)

    def set_source(self, src_rgb):
        if src_rgb is None:
            with self._lock:
                self._source = None
                self._status = "请先上传源脸照片。"
            return self._status
        try:
            face = extract_source_face(src_rgb, is_rgb=True)
            with self._lock:
                self._source = face
                emb_dim = int(getattr(face, "embedding", np.array([])).shape[0] or 0)
                self._status = (
                    f"源脸已缓存 embedding({emb_dim}-d)，摄像头循环中不再重复提取。"
                )
            return self._status
        except Exception as e:
            with self._lock:
                self._source = None
                self._status = f"源脸提取失败: {e}"
            return self._status

    def predict_stream(self, frame_rgb):
        """Non-blocking: publish latest frame or drop; never queue."""
        now = time.perf_counter()
        if self._last_cb_t > 0:
            self._cb_times.append(now - self._last_cb_t)
        self._last_cb_t = now

        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            self._dropped += 1
            return self._result, self._status

        try:
            self._received += 1
            if frame_rgb is not None:
                self._latest = frame_rgb
                self._cv.notify()
            result = self._result
            status = self._status
        finally:
            self._lock.release()
        return result, status

    def _workloop(self):
        while not self._stop:
            with self._cv:
                while self._latest is None and not self._stop:
                    self._cv.wait(timeout=0.02)
                if self._stop:
                    break
                frame = self._latest
                self._latest = None
                self._busy = True
            try:
                if frame is not None:
                    self._process_frame(frame)
            except Exception as e:
                with self._lock:
                    self._status = f"处理异常: {e}"
            finally:
                with self._lock:
                    self._busy = False
                    if self._latest is not None:
                        self._cv.notify()

    def _process_frame(self, frame_rgb: np.ndarray):
        t0 = time.perf_counter()
        small_rgb = resize_keep_aspect(frame_rgb, self.width)
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

        with self._lock:
            source = self._source

        if source is None:
            display = small_rgb
            overlay = "UPLOAD SOURCE FIRST"
            swapped_ok = False
            did_detect = False
        else:
            self._seq += 1
            seq = self._seq
            do_detect = (self._target is None) or (
                seq - self._last_det_seq >= self.interval
            )
            target = self._target
            did_detect = do_detect
            if do_detect:
                faces = detect_faces(small_bgr)
                target = None
                if faces:
                    target = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                with self._lock:
                    self._target = target
                    self._last_det_seq = seq

            if target is None:
                display = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
                overlay = "NO FACE DETECTED"
                swapped_ok = False
            else:
                # Swapper-only on reuse frames (cached Face landmarks/bbox).
                out_bgr = swap_faces(small_bgr, [target], source)
                display = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                overlay = "LIVE|det" if did_detect else "LIVE|swap"
                swapped_ok = True

        dt = time.perf_counter() - t0
        ms = dt * 1000.0
        cv2.putText(
            display,
            f"{overlay} {ms:.0f}ms",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        with self._lock:
            self._processed += 1
            if swapped_ok:
                self._swapped += 1
            self._ai_ms.append(ms)
            ai_avg = sum(self._ai_ms) / len(self._ai_ms)
            ai_fps = 1000.0 / ai_avg if ai_avg > 0 else 0.0
            if self._cb_times:
                cb_avg = sum(self._cb_times) / len(self._cb_times)
                stream_fps = 1.0 / cb_avg if cb_avg > 0 else 0.0
            else:
                stream_fps = 0.0
            slot_drop = max(0, self._received - self._processed)
            self._status = (
                f"AI {ms:.0f}ms (avg {ai_avg:.0f}ms → {ai_fps:.1f} FPS) | "
                f"流回调 ~{stream_fps:.1f} FPS | "
                f"{self.width}px det={self.det_size} 间隔={self.interval} | "
                f"处理 {self._processed} / 换脸 {self._swapped} / "
                f"槽丢 {slot_drop} / 锁丢 {self._dropped}"
            )
            self._result = display
