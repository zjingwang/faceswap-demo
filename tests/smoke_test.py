#!/usr/bin/env python3
"""Smoke / functional checks for faceswap-demo (DirectML)."""
from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        msg = fn() or "ok"
        RESULTS.append((name, True, str(msg)))
        print(f"[PASS] {name}: {msg}", flush=True)
    except Exception as e:
        RESULTS.append((name, False, f"{e}"))
        print(f"[FAIL] {name}: {e}", flush=True)
        traceback.print_exc()


def test_imports():
    import ai_core  # noqa: F401
    import web_app  # noqa: F401
    import webcam_session  # noqa: F401
    import voice_utils  # noqa: F401
    return "ai_core, web_app, webcam_session, voice_utils"


def test_providers():
    import onnxruntime as ort
    from ai_core import ORT_PROVIDERS

    avail = ort.get_available_providers()
    assert "DmlExecutionProvider" in avail, f"DML missing: {avail}"
    assert ORT_PROVIDERS[0] == "DmlExecutionProvider"
    return f"avail={avail}"


def test_load_models():
    from ai_core import ensure_models

    app, swapper = ensure_models(det_size=320)
    assert app is not None and swapper is not None
    return "FaceAnalysis + inswapper loaded"


def test_still_swap():
    import cv2
    from ai_core import detect_faces, extract_source_face, swap_faces

    src_path = os.path.join(BASE, "input", "biden.jpg")
    tgt_path = os.path.join(BASE, "input", "obama.jpg")
    assert os.path.isfile(src_path) and os.path.isfile(tgt_path)
    src_bgr = cv2.imread(src_path)
    tgt_bgr = cv2.imread(tgt_path)
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    source = extract_source_face(src_rgb, is_rgb=True)
    faces = detect_faces(tgt_bgr)
    assert faces, "no face on target"
    out = swap_faces(tgt_bgr, faces[:1], source)
    out_path = os.path.join(BASE, "output", "smoke_still.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    assert os.path.getsize(out_path) > 1000
    return f"wrote {out_path} ({os.path.getsize(out_path)} bytes)"


def test_gradio_build():
    """Construct Gradio Blocks UI without launching the server indefinitely."""
    import gradio as gr
    from webcam_session import WebcamSwap

    cam = WebcamSwap(frame_width=320, det_size=320, det_interval=3)
    with gr.Blocks(title="smoke") as demo:
        gr.Markdown("# smoke")
        with gr.Tabs():
            with gr.Tab("cam"):
                img = gr.Image(type="numpy", streaming=True, sources=["webcam"])
                out = gr.Image(type="numpy", interactive=False)
                st = gr.Textbox()
                img.stream(
                    fn=cam.predict_stream,
                    inputs=[img],
                    outputs=[out, st],
                    stream_every=0.033,
                    concurrency_limit=1,
                )
            with gr.Tab("vid"):
                gr.Video()
                gr.Button("go")
    assert demo is not None
    # queue API used by web_app
    demo.queue(default_concurrency_limit=1)
    return f"gradio Blocks OK (version={gr.__version__})"


def test_offline_video():
    import cv2
    import numpy as np
    from ai_core import extract_source_face
    from web_app import _swap_video_file

    src_path = os.path.join(BASE, "input", "biden.jpg")
    tgt_path = os.path.join(BASE, "input", "obama.jpg")
    src_bgr = cv2.imread(src_path)
    tgt_bgr = cv2.imread(tgt_path)
    source = extract_source_face(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB), is_rgb=True)

    work = tempfile.mkdtemp(prefix="smoke_vid_")
    try:
        vid_path = os.path.join(work, "tiny.mp4")
        h, w = tgt_bgr.shape[:2]
        # Keep small for speed
        frame = cv2.resize(tgt_bgr, (320, int(320 * h / w)))
        writer = cv2.VideoWriter(
            vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (frame.shape[1], frame.shape[0])
        )
        assert writer.isOpened()
        for _ in range(5):
            writer.write(frame)
        writer.release()

        silent = os.path.join(work, "silent.mp4")
        logs: list[str] = []

        def log(m):
            logs.append(m)

        n, swapped = _swap_video_file(
            vid_path,
            source,
            silent,
            ghost_mode=False,
            face_index=0,
            progress=None,
            log=log,
        )
        assert n == 5, f"expected 5 frames, got {n}"
        assert os.path.isfile(silent) and os.path.getsize(silent) > 500
        out_final = os.path.join(BASE, "output", "smoke_video.mp4")
        shutil.copy2(silent, out_final)
        return f"frames={n} swapped={swapped} -> {out_final}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()


def test_voice_mux():
    import cv2
    from voice_utils import mux_av, process_voice, replace_with_source_voice, run_ffmpeg

    work = tempfile.mkdtemp(prefix="smoke_voice_")
    try:
        # silent video via ffmpeg
        silent = os.path.join(work, "silent.mp4")
        run_ffmpeg(
            [
                "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", silent,
            ]
        )
        wav = os.path.join(work, "tone.wav")
        run_ffmpeg(
            [
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                "-ac", "1", "-ar", "16000", wav,
            ]
        )
        proc = os.path.join(work, "proc.wav")
        replace_with_source_voice(wav, 1.0, proc)
        assert os.path.isfile(proc)
        out = os.path.join(BASE, "output", "smoke_mux.mp4")
        mux_av(silent, proc, out)
        assert os.path.isfile(out) and os.path.getsize(out) > 500
        # process_voice path
        proc2 = os.path.join(work, "proc2.wav")
        process_voice("", wav, proc2, video_path=silent)
        return f"mux ok -> {out} ({os.path.getsize(out)} bytes)"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    check("imports", test_imports)
    check("providers", test_providers)
    check("load_models", test_load_models)
    check("still_swap", test_still_swap)
    check("gradio_build", test_gradio_build)
    check("offline_video", test_offline_video)
    check("voice_mux", test_voice_mux)

    print("\n=== SUMMARY ===")
    failed = 0
    for name, ok, msg in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}: {name} — {msg}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
