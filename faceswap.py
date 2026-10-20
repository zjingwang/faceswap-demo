#!/usr/bin/env python3
"""Minimum CPU face swap (Linux/无显卡虚拟机可用).

用法:
    python faceswap.py --source 人脸源图.jpg --target 目标图.jpg --output 结果.png

说明:
    - 全部资源在当前目录: 模型在 ./models, 结果由 --output 指定。
    - 纯 CPU 推理 (CPUExecutionProvider)。
    - 不要开任何 "面部修复" 环节, 这是专为纯 CPU 准备的极简版本。
"""
import argparse
import os
import sys

import cv2
import insightface
from insightface.app import FaceAnalysis

INSWAPPER_MODEL = "inswapper_128.onnx"


def main() -> int:
    ap = argparse.ArgumentParser(description="纯 CPU 照片换脸 (inswapper_128)")
    ap.add_argument("--source", required=True, help="人脸来源图(提供脸)")
    ap.add_argument("--target", required=True, help="目标图(被换脸)")
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "output.png"),
                    help="输出图片路径")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "models"),
                    help="放置 inswapper_128.onnx 的目录")
    ap.add_argument("--root", default=os.path.dirname(__file__),
                    help="FaceAnalysis 根目录(buffalo_l 放在 <root>/models/buffalo_l)")
    ap.add_argument("--det-size", type=int, default=640, help="检测分辨率, 越小越快, 建议 320-640")
    args = ap.parse_args()

    # 0) 检查 inswapper 模型是否已就位
    swapper_path = os.path.join(args.models, INSWAPPER_MODEL)
    if not os.path.isfile(swapper_path):
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
        sys.exit(f"缺少模型 {swapper_path}\n请先手动下载(约550MB):\n  curl -L -o {swapper_path} {url}")

    # 1) 人脸分析器 (buffalo_l: 检测+关键点+特征, 首次运行会在 <root>/models 下载)
    print("[1/4] 加载人脸分析器 buffalo_l ...", flush=True)
    app = FaceAnalysis(name="buffalo_l", root=args.root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # 2) 加载换脸器
    print("[2/4] 加载换脸器 inswapper_128 ...", flush=True)
    swapper = insightface.model_zoo.get_model(swapper_path, providers=["CPUExecutionProvider"])

    # 3) 读图 + 提脸
    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    if src is None:
        sys.exit(f"无法读取源图: {args.source}")
    if tgt is None:
        sys.exit(f"无法读取目标图: {args.target}")

    print("[3/4] 识别人脸 ...", flush=True)
    src_faces = app.get(src)
    tgt_faces = app.get(tgt)
    if not src_faces:
        sys.exit("源图里没检测到人脸")
    if not tgt_faces:
        sys.exit("目标图里没检测到人脸")

    src_face = src_faces[0]
    # 优先替换目标图里最大的人脸
    tgt_faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    target_face = tgt_faces[0]

    # 4) 换脸
    print("[4/4] 开始换脸 (纯 CPU) ...", flush=True)
    result = swapper.get(tgt, target_face, src_face, paste_back=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"完成 -> {os.path.abspath(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
