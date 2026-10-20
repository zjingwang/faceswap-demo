#!/usr/bin/env python3
"""下载 faceswap-demo 所需模型 (buffalo_l 模型包 + inswapper_128.onnx)。

用法:
    python download_models.py                # 下载到项目 ./models 目录
    python download_models.py --out /path/to/models

说明:
    - buffalo_l:      insightface 人脸分析模型包 (检测 + 关键点 + 特征), 约 340MB
    - inswapper_128.onnx:  换脸模型, 约 550MB
    - 脚本可重复执行: 已存在的文件会自动跳过, 可在中断后续传/补全。
    - 模型较大 (合计约 0.9GB), 请在有网络的环境中运行。
"""
import argparse
import os
import sys
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE, "models")

INSWAPPER_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# insightface buffalo_l 包内需要的 5 个文件
BUFFALO_L_FILES = ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx",
                   "1k3d68.onnx", "genderage.onnx")


def _progress(done: int, total: int) -> None:
    if total:
        pct = done / total * 100
        print(f"\r  已下载 {done / 1048576:.1f} / {total / 1048576:.1f} MB ({pct:.1f}%)",
              end="", flush=True)
    else:
        print(f"\r  已下载 {done / 1048576:.1f} MB", end="", flush=True)


def download(url: str, path: str, chunk: int = 1024 * 1024) -> None:
    print(f"[1/2] 下载 {url}")
    print(f"      -> {path}")
    req = urllib.request.Request(url, headers={"User-Agent": "faceswap-demo/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(path, "wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                _progress(done, total)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 faceswap-demo 所需模型")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"模型目录 (默认 {DEFAULT_OUT})")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    buf_dir = os.path.join(out_dir, "buffalo_l")

    # 1) inswapper_128.onnx
    swapper = os.path.join(out_dir, "inswapper_128.onnx")
    if os.path.isfile(swapper):
        print(f"[跳过] 已存在 {swapper}")
    else:
        download(INSWAPPER_URL, swapper)

    # 2) buffalo_l 模型包
    missing = [f for f in BUFFALO_L_FILES
               if not os.path.isfile(os.path.join(buf_dir, f))]
    if not missing:
        print(f"[跳过] buffalo_l 已完整存在 ({buf_dir})")
    else:
        zpath = os.path.join(out_dir, "buffalo_l.zip")
        download(BUFFALO_L_URL, zpath)
        print(f"[2/2] 解压 {zpath} -> {out_dir}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)
        os.remove(zpath)
        # 校验
        still_missing = [f for f in BUFFALO_L_FILES
                         if not os.path.isfile(os.path.join(buf_dir, f))]
        if still_missing:
            sys.exit(f"[错误] buffalo_l 解压后仍缺少: {still_missing}")

    print("\n完成! 现在可以运行:")
    print("  python faceswap.py --source input/biden.jpg --target input/obama.jpg --output output/demo.png")
    print("  python web_app.py            # 打开 http://127.0.0.1:7860")
    return 0


if __name__ == "__main__":
    sys.exit(main())
