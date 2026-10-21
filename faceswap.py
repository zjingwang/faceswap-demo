#!/usr/bin/env python3
"""命令行照片换脸 — AMD DirectML (onnxruntime-directml), 无 CUDA。

用法:
    python faceswap.py --source 人脸源图.jpg --target 目标图.jpg --output 结果.png
"""
import argparse
import os
import sys

import cv2

from ai_core import ORT_PROVIDERS, detect_faces, ensure_models, swap_faces

INSWAPPER_MODEL = "inswapper_128.onnx"


def main() -> int:
    ap = argparse.ArgumentParser(description="AMD DirectML 照片换脸 (inswapper_128)")
    ap.add_argument("--source", required=True, help="人脸来源图(提供脸)")
    ap.add_argument("--target", required=True, help="目标图(被换脸)")
    ap.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "output.png"),
        help="输出图片路径",
    )
    ap.add_argument(
        "--models",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="放置 inswapper_128.onnx 的目录",
    )
    ap.add_argument("--det-size", type=int, default=640, help="检测分辨率, 越小越快")
    args = ap.parse_args()

    swapper_path = os.path.join(args.models, INSWAPPER_MODEL)
    if not os.path.isfile(swapper_path):
        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
        sys.exit(f"缺少模型 {swapper_path}\n请运行 python download_models.py\n或手动下载:\n  {url}")

    print(f"[1/4] providers = {ORT_PROVIDERS}", flush=True)
    ensure_models(det_size=args.det_size)
    print("[2/4] 模型已加载", flush=True)

    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    if src is None:
        sys.exit(f"无法读取源图: {args.source}")
    if tgt is None:
        sys.exit(f"无法读取目标图: {args.target}")

    print("[3/4] 识别人脸 ...", flush=True)
    src_faces = detect_faces(src)
    tgt_faces = detect_faces(tgt)
    if not src_faces:
        sys.exit("源图里没检测到人脸")
    if not tgt_faces:
        sys.exit("目标图里没检测到人脸")

    src_face = src_faces[0]
    tgt_faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True
    )
    target_face = tgt_faces[0]

    print("[4/4] 开始换脸 (DirectML) ...", flush=True)
    result = swap_faces(tgt, [target_face], src_face)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"完成 -> {os.path.abspath(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
