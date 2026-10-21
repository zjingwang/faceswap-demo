#!/usr/bin/env python3
"""AMD DirectML face-swap Gradio platform.

Tab1: Webcam live swap (low-res AI, cached embedding, drop-frame)
Tab2: Offline video face swap
Tab3: Dual-column ultimate fusion (face + best-effort voice)

Hardware: Ryzen + Radeon 780M shared RAM. Providers: Dml + CPU. No CUDA.

Launch:
    python app.py
    python web_app.py [--port 7860]
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np
from tqdm import tqdm

import ai_core
from ai_core import (
    ORT_PROVIDERS,
    detect_faces,
    ensure_models,
    extract_source_face,
    resize_keep_aspect,
    sort_faces_left_to_right,
    stretch_to,
    swap_faces,
)
from voice_utils import (
    extract_audio,
    make_work_dir,
    mux_av,
    process_voice,
    probe_duration_seconds,
)
from live_capture import LiveCaptureSession

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
SWAPPER_PATH = os.path.join(BASE, "models", "inswapper_128.onnx")
VIDEO_AI_WIDTH = 640
VIDEO_DET_SIZE = 320
# Restored after offline/fusion so Tab1 keeps its lighter det_size.
_ACTIVE_WEBCAM_DET = 160
# How often Gradio pulls the latest frame from the OpenCV capture thread.
LIVE_UI_REFRESH = 0.12


def _restore_webcam_det() -> None:
    try:
        ai_core.set_det_size(_ACTIVE_WEBCAM_DET)
    except Exception:
        pass


def _model_banner() -> str:
    if os.path.isfile(SWAPPER_PATH):
        return f"models OK · providers=`{ORT_PROVIDERS}`"
    return (
        f"models MISSING — 请运行 `python download_models.py` "
        f"（期望 `{SWAPPER_PATH}`）"
    )


def _pick_faces_for_frame(faces, ghost_mode: bool, face_index: int, locked_order):
    if not faces:
        return [], locked_order
    if ghost_mode:
        return faces, locked_order

    ordered = sort_faces_left_to_right(faces)
    if locked_order is None:
        locked_order = [
            ((f.bbox[0] + f.bbox[2]) * 0.5, (f.bbox[1] + f.bbox[3]) * 0.5)
            for f in ordered
        ]

    idx = int(face_index)
    if idx < 0 or idx >= len(locked_order):
        return [], locked_order

    tx, ty = locked_order[idx]
    best = min(
        faces,
        key=lambda f: (
            ((f.bbox[0] + f.bbox[2]) * 0.5 - tx) ** 2
            + ((f.bbox[1] + f.bbox[3]) * 0.5 - ty) ** 2
        ),
    )
    return [best], locked_order


def _swap_video_file(
    video_path: str,
    source_face,
    silent_mp4: str,
    *,
    ghost_mode: bool,
    face_index: int,
    progress,
    log,
    progress_scale: float = 0.85,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 约 {total} 帧；AI 宽={VIDEO_AI_WIDTH}")

    writer = cv2.VideoWriter(
        silent_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建 VideoWriter")

    locked_order = None
    frame_i = 0
    swapped_frames = 0
    pbar = tqdm(total=total if total > 0 else None, desc="换脸", disable=False)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = resize_keep_aspect(frame, VIDEO_AI_WIDTH)
            faces = detect_faces(small)
            if frame_i == 0:
                n = len(faces)
                if ghost_mode:
                    log(f"第 1 帧检测到 {n} 张人脸，全员鬼畜模式：替换全部人脸")
                else:
                    ordered = sort_faces_left_to_right(faces)
                    log(
                        f"第 1 帧检测到 {n} 张人脸，按左→右排序，"
                        f"仅替换编号 {int(face_index)}（可选 {len(ordered)}）"
                    )
            to_swap, locked_order = _pick_faces_for_frame(
                faces, bool(ghost_mode), int(face_index or 0), locked_order
            )
            if to_swap:
                small = swap_faces(small, to_swap, source_face)
                swapped_frames += 1
            out = stretch_to(small, (width, height))
            writer.write(out)
            frame_i += 1
            pbar.update(1)
            if progress is not None and total > 0:
                progress(
                    min(progress_scale, progress_scale * frame_i / total),
                    desc=f"换脸 {frame_i}/{total}",
                )
            del frame, small, out
    finally:
        pbar.close()
        writer.release()
        cap.release()
        gc.collect()
        log("已 cap.release() + gc.collect()，释放共享 GPU/内存缓存。")

    log(f"画面换脸完成：{frame_i} 帧，成功换脸 {swapped_frames} 帧")
    return frame_i, swapped_frames


def offline_video_swap(source_img, video, progress=gr.Progress(track_tqdm=True)):
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        logs.append(line)
        print(line, flush=True)

    if source_img is None:
        log("请上传源脸照片。")
        return None, "\n".join(logs)
    if not video:
        log("请上传 MP4 视频。")
        return None, "\n".join(logs)

    video_path = video if isinstance(video, str) else getattr(video, "name", None)
    if not video_path or not os.path.isfile(video_path):
        log(f"无法读取视频: {video}")
        return None, "\n".join(logs)

    try:
        ensure_models(det_size=VIDEO_DET_SIZE)
        ai_core.set_det_size(VIDEO_DET_SIZE)
        log(f"providers = {ORT_PROVIDERS}")
        source_face = extract_source_face(source_img, is_rgb=True)
        log("源脸 embedding 已提取（一次）。")
    except Exception as e:
        log(f"模型/源脸失败: {e}")
        return None, "\n".join(logs)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = make_work_dir("offline_")
    silent = os.path.join(work, "silent.mp4")
    final = os.path.join(OUT_DIR, f"video_swap_{stamp}.mp4")

    try:
        _swap_video_file(
            video_path,
            source_face,
            silent,
            ghost_mode=False,
            face_index=0,
            progress=progress,
            log=log,
            progress_scale=0.95,
        )
        raw = os.path.join(work, "audio.wav")
        if extract_audio(video_path, raw, log=log):
            mux_av(silent, raw, final, log=log)
        else:
            shutil.copy2(silent, final)
            log("无原音轨，输出无声换脸视频。")
        if progress is not None:
            progress(1.0, desc="完成")
        log(f"完成 -> {final}")
        return final, "\n".join(logs)
    except Exception as e:
        log(f"失败: {e}")
        return None, "\n".join(logs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
        _restore_webcam_det()


def ultimate_fusion(
    source_img,
    source_voice,
    target_video,
    ghost_mode,
    face_index,
    progress=gr.Progress(track_tqdm=True),
):
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        logs.append(line)
        print(line, flush=True)

    if source_img is None:
        log("请上传清晰正脸源照片。")
        return None, "\n".join(logs)
    if not target_video:
        log("请上传目标视频。")
        return None, "\n".join(logs)

    video_path = (
        target_video
        if isinstance(target_video, str)
        else getattr(target_video, "name", None)
    )
    if not video_path or not os.path.isfile(video_path):
        log(f"无法读取目标视频: {target_video}")
        return None, "\n".join(logs)

    voice_path = None
    if source_voice is not None:
        voice_path = (
            source_voice
            if isinstance(source_voice, str)
            else getattr(source_voice, "name", None)
        )

    try:
        ensure_models(det_size=VIDEO_DET_SIZE)
        ai_core.set_det_size(VIDEO_DET_SIZE)
        log(f"providers = {ORT_PROVIDERS}")
        log("提取源脸 embedding（仅一次）…")
        source_face = extract_source_face(source_img, is_rgb=True)
        log("源脸就绪。")
    except Exception as e:
        log(f"模型/源脸失败: {e}")
        return None, "\n".join(logs)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = make_work_dir("fusion_")
    silent = os.path.join(work, "swapped_silent.mp4")
    final = os.path.join(OUT_DIR, f"fusion_final_{stamp}.mp4")

    mode = "全员鬼畜模式" if ghost_mode else f"精准目标人脸编号={int(face_index or 0)}"
    log(f"人脸过滤: {mode}")

    try:
        _swap_video_file(
            video_path,
            source_face,
            silent,
            ghost_mode=bool(ghost_mode),
            face_index=int(face_index or 0),
            progress=progress,
            log=log,
            progress_scale=0.8,
        )

        raw = os.path.join(work, "target_audio.wav")
        proc = os.path.join(work, "final_audio.wav")
        has_audio = extract_audio(video_path, raw, log=log)
        if progress is not None:
            progress(0.88, desc="处理人声")

        duration = probe_duration_seconds(video_path)
        if voice_path and os.path.isfile(voice_path):
            process_voice(
                raw if has_audio else "",
                voice_path,
                proc,
                video_path=video_path,
                log=log,
            )
            if progress is not None:
                progress(0.95, desc="音视频封装")
            mux_av(silent, proc, final, log=log)
        elif has_audio:
            log("未上传源声音，保留目标视频原音轨。")
            mux_av(silent, raw, final, log=log)
        else:
            log("无源声音且目标无音轨，输出无声换脸视频。")
            shutil.copy2(silent, final)

        if progress is not None:
            progress(1.0, desc="完成")
        log(f"究极融合完成 -> {final} (duration≈{duration:.2f}s)")
        return final, "\n".join(logs)
    except Exception as e:
        log(f"失败: {e}；尝试返回无声换脸结果。")
        try:
            shutil.copy2(silent, final)
            return final, "\n".join(logs)
        except Exception:
            return None, "\n".join(logs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
        _restore_webcam_det()


def main() -> int:
    ap = argparse.ArgumentParser(description="AMD DirectML face-swap platform")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--det-size", type=int, default=160)
    ap.add_argument(
        "--ui-refresh",
        type=float,
        default=LIVE_UI_REFRESH,
        help="Gradio preview poll interval (OpenCV runs in background)",
    )
    args = ap.parse_args()

    global _ACTIVE_WEBCAM_DET
    _ACTIVE_WEBCAM_DET = int(args.det_size)
    ui_refresh = max(0.05, float(args.ui_refresh))

    print(_model_banner(), flush=True)
    print(
        f"[web_app] live: OpenCV background capture → Gradio preview; "
        f"width={args.width} det={args.det_size} ui_refresh={ui_refresh}s",
        flush=True,
    )
    live = LiveCaptureSession(width=args.width, det_size=args.det_size)
    try:
        ensure_models(det_size=args.det_size, live=True)
    except Exception as e:
        print(f"[web_app] 模型预热失败（可在 UI 中重试）: {e}", flush=True)

    with gr.Blocks(title="AMD Face Swap (DirectML)") as demo:
        gr.Markdown(
            f"# AMD 本地换脸平台\n"
            f"**{_model_banner()}** — Ryzen / Radeon 780M · `onnxruntime-directml` · 无 CUDA"
        )

        with gr.Tabs():
            with gr.Tab("① 实时换脸"):
                gr.Markdown(
                    "本机 **OpenCV 后台采摄像头 + 换脸**，画面显示在本页预览里"
                    "（**不弹独立窗口**）。默认宽 "
                    f"**{args.width}** / det **{args.det_size}**。"
                )
                with gr.Row():
                    with gr.Column():
                        src_cam = gr.Image(
                            label="源脸照片（缓存 embedding）",
                            type="numpy",
                            sources=["upload"],
                            height=280,
                            format="jpeg",
                        )
                        width_sl = gr.Slider(
                            160,
                            400,
                            value=args.width,
                            step=8,
                            label="处理宽度（越小越流畅）",
                        )
                        det = gr.Radio(
                            [160, 192, 256, 320],
                            value=args.det_size,
                            label="检测分辨率",
                        )
                        vcam_ck = gr.Checkbox(
                            label="输出到 OBS 虚拟摄像头（给微信用）",
                            value=True,
                            info="需已安装 OBS Studio；微信里选摄像头「OBS Virtual Camera」",
                        )
                        with gr.Row():
                            start_btn = gr.Button("▶ 开始实时换脸", variant="primary")
                            stop_btn = gr.Button("⏹ 停止")
                        help_box = gr.Markdown(
                            "### 微信视频怎么用换脸\n"
                            "1. 本机已装 **OBS Studio**（虚拟摄像头驱动）\n"
                            "2. 勾选「输出到 OBS 虚拟摄像头」→ 点开始\n"
                            "3. 微信视频通话 → 摄像头选 **OBS Virtual Camera**\n"
                            "4. 不要用微信直接选物理摄像头，否则对方看到的是真脸\n"
                            "5. 若虚拟摄像头打不开：先打开一次 OBS，点「启动虚拟摄像头」再关 OBS，然后重试"
                        )
                    with gr.Column():
                        preview = gr.Image(
                            label="实时换脸预览（本机摄像头）",
                            type="numpy",
                            interactive=False,
                            height=420,
                            format="jpeg",
                        )
                status_cam = gr.Textbox(label="状态", lines=2)

                def change_width(v):
                    live.width = max(128, int(v))
                    return f"处理宽度={live.width}"

                def change_det(v):
                    global _ACTIVE_WEBCAM_DET
                    live.det_size = int(v)
                    _ACTIVE_WEBCAM_DET = int(v)
                    ai_core.set_det_size(int(v))
                    return f"检测分辨率={v}"

                def start_live(use_vcam):
                    msg = live.start(-1, virtual_cam=bool(use_vcam))
                    return msg, live.get_preview()

                def stop_live():
                    msg = live.stop()
                    return msg, live.get_preview()

                def refresh_ui():
                    return live.get_preview(), live.get_status()

                src_cam.change(
                    fn=live.set_source, inputs=[src_cam], outputs=[status_cam]
                )
                width_sl.change(
                    fn=change_width, inputs=[width_sl], outputs=[status_cam]
                )
                det.change(fn=change_det, inputs=[det], outputs=[status_cam])
                start_btn.click(
                    fn=start_live,
                    inputs=[vcam_ck],
                    outputs=[status_cam, preview],
                )
                stop_btn.click(fn=stop_live, outputs=[status_cam, preview])
                ui_timer = gr.Timer(ui_refresh)
                ui_timer.tick(fn=refresh_ui, outputs=[preview, status_cam])

            with gr.Tab("② 离线视频换脸"):
                gr.Markdown(
                    "上传源脸 + MP4，逐帧处理（内存仅 1 帧），进度 0–100%。保留原音轨（若有）。"
                )
                with gr.Row():
                    v_src = gr.Image(label="源脸照片", type="numpy", height=280)
                    v_vid = gr.Video(label="目标 MP4", height=280)
                v_btn = gr.Button("开始离线视频换脸", variant="primary")
                v_out = gr.Video(label="换脸结果")
                v_log = gr.Textbox(label="日志", lines=10)
                v_btn.click(
                    offline_video_swap, inputs=[v_src, v_vid], outputs=[v_out, v_log]
                )

            with gr.Tab("③ 究极融合（脸+声）"):
                gr.Markdown(
                    "双栏布局：逐帧换脸 + 最佳努力本地人声替换"
                    "（ffmpeg 循环填充，非神经克隆）。"
                )
                with gr.Row():
                    with gr.Column():
                        f_face = gr.Image(
                            label="清晰正脸源照片", type="numpy", height=220
                        )
                        f_voice = gr.Audio(
                            label="源声音片段（voice cloning 输入）",
                            type="filepath",
                        )
                        f_video = gr.Video(label="目标视频（舞蹈/电影等）", height=220)
                        f_all = gr.Checkbox(label="全员鬼畜模式", value=False)
                        f_idx = gr.Number(
                            label="精准目标人脸编号", value=0, precision=0
                        )
                        f_btn = gr.Button(
                            "🚀 启动究极融合（本地 AMD 加速）",
                            variant="primary",
                        )
                    with gr.Column():
                        f_out = gr.Video(label="最终视频（脸+声）")
                        f_log = gr.Textbox(label="详细日志流", lines=18)
                f_btn.click(
                    ultimate_fusion,
                    inputs=[f_face, f_voice, f_video, f_all, f_idx],
                    outputs=[f_out, f_log],
                )

        gr.Markdown(
            "模型路径：`models/inswapper_128.onnx` + `models/buffalo_l/` · "
            "Tab③ 需要系统 PATH 中的 **ffmpeg**。"
        )

    # Keep overall concurrency modest; webcam stream already uses concurrency_limit=1.
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )
    live.stop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止")
