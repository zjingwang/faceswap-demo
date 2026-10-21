#!/usr/bin/env python3
"""Local OpenCV realtime face-swap (DirectML).

- Auto-skips black/dummy cameras (often index 0 = OBS Virtual Cam)
- Window X button and q both exit cleanly
- Writes PID file so Gradio can stop the process

Keys: q=quit  s=toggle  [/]=width  c=next camera
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import time

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

WIN = "AMD Live FaceSwap (q=quit)"
PID_FILE = os.path.join(os.environ.get("TEMP", "."), "faceswap_live_local.pid")


def _frame_ok(frame) -> bool:
    if frame is None or frame.size == 0:
        return False
    return float(frame.mean()) > 8.0


def _write_pid():
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _clear_pid():
    try:
        if os.path.isfile(PID_FILE):
            os.remove(PID_FILE)
    except OSError:
        pass


def open_camera(index: int):
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
            return cap, frame
    if frame is not None and frame.size:
        return cap, frame
    cap.release()
    return None


def find_camera(preferred: int | None = None):
    order = []
    if preferred is not None and preferred >= 0:
        order.append(preferred)
    for i in range(0, 6):
        if i not in order:
            order.append(i)
    black_hits = []
    for i in order:
        opened = open_camera(i)
        if opened is None:
            continue
        cap, frame = opened
        if _frame_ok(frame):
            print(f"[live] camera index={i} mean={frame.mean():.1f}", flush=True)
            return cap, i, frame
        print(f"[live] skip index={i} (black mean={frame.mean():.1f})", flush=True)
        black_hits.append(i)
        cap.release()
    raise RuntimeError(
        "无可用摄像头。请先关闭 Gradio 页内摄像头后再试。"
        f" 全黑设备 index={black_hits}"
    )


def window_closed() -> bool:
    try:
        # <1 means user clicked the window X
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", type=int, default=-1)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--det-size", type=int, default=160)
    ap.add_argument("--interval", type=int, default=1)
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        print(f"源脸不存在: {args.source}", file=sys.stderr)
        return 1

    _write_pid()
    atexit.register(_clear_pid)

    print(f"providers={ORT_PROVIDERS}", flush=True)
    print("请先关闭浏览器 Gradio 摄像头，避免抢占设备。", flush=True)
    ensure_models(det_size=args.det_size, live=True)
    src_bgr = cv2.imread(args.source)
    if src_bgr is None:
        print("无法读取源脸", file=sys.stderr)
        return 1
    source = extract_source_face(src_bgr, is_rgb=False)

    preferred = None if args.camera < 0 else args.camera
    try:
        cap, cam_idx, _ = find_camera(preferred)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    width = max(128, int(args.width))
    interval = max(1, int(args.interval))
    enable_swap = True
    last_swapped = None
    seq = 0
    ai_ms = 0.0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 640, 480)
    print("q/关窗退出 | s 开关 | [ ] 宽度 | c 换摄像头", flush=True)

    try:
        while True:
            if window_closed():
                break
            ok, frame = cap.read()
            if not ok:
                break
            seq += 1

            if not _frame_ok(frame):
                show = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    show,
                    f"BLACK cam={cam_idx}  press c / close Gradio cam",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                show = frame
                if enable_swap and (seq % interval == 0):
                    t0 = time.perf_counter()
                    small = resize_keep_aspect(frame, width)
                    faces = detect_faces(small)
                    if faces:
                        target = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0])
                            * (f.bbox[3] - f.bbox[1]),
                        )
                        small = swap_faces(small, [target], source)
                        h, w = frame.shape[:2]
                        last_swapped = cv2.resize(
                            small, (w, h), interpolation=cv2.INTER_LINEAR
                        )
                        show = last_swapped
                    ai_ms = (time.perf_counter() - t0) * 1000.0
                elif enable_swap and last_swapped is not None:
                    show = last_swapped

                ai_fps = 1000.0 / ai_ms if ai_ms > 0 else 0.0
                cv2.putText(
                    show,
                    f"{'ON' if enable_swap else 'OFF'} {ai_ms:.0f}ms "
                    f"~{ai_fps:.1f}FPS cam={cam_idx} w={width}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(WIN, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("s"):
                enable_swap = not enable_swap
            if key == ord("["):
                width = max(128, width - 32)
            if key == ord("]"):
                width = min(640, width + 32)
            if key == ord("c"):
                cap.release()
                try:
                    cap, cam_idx, _ = find_camera((cam_idx + 1) % 6)
                    last_swapped = None
                except RuntimeError as e:
                    print(str(e), flush=True)
                    break
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # Pump GUI events so window actually disappears on Windows
        for _ in range(5):
            cv2.waitKey(1)
        _clear_pid()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
