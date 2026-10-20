#!/usr/bin/env python3
"""本地照片换脸 Web 界面 (Gradio / 纯 CPU / 无显卡可用).

启动:
    ./venv/bin/python web_app.py [--port 7860]
然后浏览器打开:  http://127.0.0.1:7860

说明:
    - 只用 CPU(CPUExecutionProvider), 适合无独显的机器/虚拟机。
    - 模型只在启动后第一次使用时加载(~10秒), 之后每次换脸约几秒。
    - 不开启任何"面部修复"环节(GFPGAN/CodeFormer 那种很吃资源)。
    - 服务只绑定本机 127.0.0.1, 不对外网开放。
    - 每次换脸会把 结果图 和 对比图 另存到 output/ 目录。
"""
import argparse
import os
import sys
import time

# 本地 Demo, 不向 Gradio 上报使用统计
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
SWAPPER_PATH = os.path.join(BASE, "models", "inswapper_128.onnx")

_app = None
_swapper = None


def get_models():
    """模型懒加载, 全局只需一次。"""
    global _app, _swapper
    if _app is not None:
        return _app, _swapper
    if not os.path.isfile(SWAPPER_PATH):
        raise RuntimeError(f"缺少模型 {SWAPPER_PATH}, 请先跑过 faceswap.py 或手动下载")
    print("[web_app] 加载 buffalo_l ...", flush=True)
    _app = FaceAnalysis(name="buffalo_l", root=BASE, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))
    print("[web_app] 加载 inswapper_128 ...", flush=True)
    _swapper = insightface.model_zoo.get_model(SWAPPER_PATH, providers=["CPUExecutionProvider"])
    print("[web_app] 模型就绪", flush=True)
    return _app, _swapper


def _make_comparison(src_bgr, tgt_bgr, res_bgr, max_h=640):
    """生成 源|目标|结果 三连对比图(BGR)。"""
    def rz(x):
        h = max_h
        return cv2.resize(x, (max(1, int(x.shape[1] * h / x.shape[0])), h))
    a, b, c = rz(src_bgr), rz(tgt_bgr), rz(res_bgr)
    for im, t in ((a, "SOURCE 源脸"), (b, "TARGET 目标"), (c, "RESULT 结果")):
        cv2.putText(im, t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    h = max(a.shape[0], b.shape[0], c.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.hconcat([pad(a), pad(b), pad(c)])


def process(source_rgb, target_rgb, det_size=640):
    """Gradio 回调: 输入 RGB 图, 输出 (结果RGB, 对比RGB, 状态文本)。"""
    t0 = time.time()
    if source_rgb is None or target_rgb is None:
        return None, None, "请先上传源脸照片和目标照片"
    try:
        app, swapper = get_models()
    except Exception as e:
        return None, None, f"模型加载失败: {e}"

    try:
        src = cv2.cvtColor(np.asarray(source_rgb), cv2.COLOR_RGB2BGR)
        tgt = cv2.cvtColor(np.asarray(target_rgb), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, f"图片解析失败: {e}"

    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    t1 = time.time()
    src_faces = app.get(src)
    tgt_faces = app.get(tgt)
    if not src_faces:
        return None, None, "源脸照片里没检测到人脸, 换一张脸更清楚/更居中的照片"
    if not tgt_faces:
        return None, None, "目标照片里没检测到人脸(注意: 戴口罩/侧面/太暗可能检测不到)"

    src_face = src_faces[0]
    tgt_faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    target_face = tgt_faces[0]

    t2 = time.time()
    result = swapper.get(tgt, target_face, src_face, paste_back=True)
    elapsed = time.time() - t0

    # 存一份到 output/
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res_path = os.path.join(OUT_DIR, f"web_result_{stamp}.png")
    cmp_path = os.path.join(OUT_DIR, f"web_compare_{stamp}.jpg")
    cv2.imwrite(res_path, result)
    comparison = _make_comparison(src, tgt, result)
    cv2.imwrite(cmp_path, comparison)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    comp_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
    status = (f"完成, 共 {elapsed:.1f} 秒 (含模型加载则首次更慢). "
              f"已保存: {res_path} / {cmp_path}")
    print(f"[web_app] {time.strftime('%H:%M:%S')} 换脸成功 {elapsed:.1f}s", flush=True)
    return result_rgb, comp_rgb, status


def main() -> int:
    ap = argparse.ArgumentParser(description="本地照片换脸 Web 界面")
    ap.add_argument("--port", type=int, default=7860, help="监听端口(默认 7860)")
    args = ap.parse_args()

    import gradio as gr

    with gr.Blocks(title="本地照片换脸 (纯CPU)") as demo:
        gr.Markdown(
            "## 😀 本地照片换脸\n"
            "纯 CPU 推理, 运行在你自己的机器上, 图片不会上传到任何服务器。"
        )
        with gr.Row():
            src = gr.Image(label="① 源脸照片 (提供脸)", type="numpy", height=360)
            tgt = gr.Image(label="② 目标照片 (被换脸)", type="numpy", height=360)
        with gr.Row():
            det_size = gr.Radio([640, 480, 320], value=640, label="人脸检测分辨率 (越大越准、越慢)")
            btn = gr.Button("🔄 开始换脸", variant="primary", size="lg")
        with gr.Row():
            result = gr.Image(label="换脸结果", type="numpy", height=420, interactive=False)
            compare = gr.Image(label="对比 (源 | 目标 | 结果)", type="numpy", height=420, interactive=False)
        status = gr.Textbox(label="状态", lines=3)
        gr.Markdown(
            "- 多张人脸的图: 源图取**第 1 张**, 目标图换**最大**的那张\n"
            "- 戴口罩 / 侧面 / 过暗的脸会检测不到, 属预期行为\n"
            "- 别开「面部修复」——那是给有显卡的机器用的, 你这台纯 CPU 开它会卡死\n"
            "- 每次换脸结果同时另存到项目 `output/` 目录"
        )
        btn.click(process, inputs=[src, tgt, det_size], outputs=[result, compare, status])

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        show_error=True,
        quiet=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止")
