#!/usr/bin/env python3
"""Webcam-only entry. Prefer full platform: python app.py"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from ai_core import ORT_PROVIDERS, ensure_models, set_det_size
from webcam_session import WebcamSwap


def main() -> int:
    ap = argparse.ArgumentParser(description="AMD DirectML 摄像头实时换脸")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--det-size", type=int, default=192)
    ap.add_argument("--det-interval", type=int, default=3)
    ap.add_argument("--stream-every", type=float, default=0.033)
    args = ap.parse_args()
    stream_every = max(0.016, float(args.stream_every))

    cam = WebcamSwap(
        frame_width=args.width, det_size=args.det_size, det_interval=args.det_interval
    )
    cam.start()

    import gradio as gr

    cam_opts = gr.WebcamOptions(
        mirror=True,
        constraints={"video": {"frameRate": {"ideal": 30, "max": 60}}},
    )

    with gr.Blocks(title="AMD DirectML 实时换脸") as demo:
        gr.Markdown(
            f"## 摄像头实时换脸\n"
            f"providers = `{ORT_PROVIDERS}` · stream_every={stream_every}s "
            f"(~{1.0 / stream_every:.0f} Hz)。完整平台请运行 `python app.py`。"
        )
        with gr.Row():
            with gr.Column():
                src = gr.Image(
                    label="源脸照片",
                    type="numpy",
                    sources=["upload"],
                    height=300,
                    format="jpeg",
                )
                width_sl = gr.Slider(
                    160, 320, value=args.width, step=8, label="处理宽度"
                )
                det = gr.Radio(
                    [160, 192, 256, 320], value=args.det_size, label="检测分辨率"
                )
                interval = gr.Slider(
                    1, 10, value=args.det_interval, step=1, label="跳帧间隔"
                )
            with gr.Column():
                webcam = gr.Image(
                    label="摄像头",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    height=300,
                    format="jpeg",
                    webcam_options=cam_opts,
                )
        preview = gr.Image(
            label="预览",
            type="numpy",
            interactive=False,
            streaming=True,
            height=360,
            format="jpeg",
        )
        status = gr.Textbox(label="状态", lines=3)

        def change_width(v):
            cam.width = max(128, int(v))
            return f"处理宽度已设为 {cam.width}"

        def change_det(v):
            cam.det_size = int(v)
            set_det_size(int(v))
            return f"检测分辨率已设为 {v}"

        def change_interval(v):
            cam.interval = max(1, int(v))
            return f"跳帧间隔已设为 {cam.interval}"

        webcam.stream(
            fn=cam.predict_stream,
            inputs=[webcam],
            outputs=[preview, status],
            stream_every=stream_every,
            concurrency_limit=8,
            trigger_mode="always_last",
            time_limit=None,
        )
        src.change(fn=cam.set_source, inputs=[src], outputs=[status])
        width_sl.change(fn=change_width, inputs=[width_sl], outputs=[status])
        det.change(fn=change_det, inputs=[det], outputs=[status])
        interval.change(fn=change_interval, inputs=[interval], outputs=[status])

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        show_error=True,
        quiet=True,
    )
    cam.stop()
    return 0


if __name__ == "__main__":
    try:
        ensure_models(det_size=192)
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止")
