#!/usr/bin/env python3
"""纯 CPU 实时(丢帧保时效)换脸 Web 界面 —— 基于 Gradio 摄像头流 + inswapper_128.

启动:
    python webcam_realtime.py [--port 7861]

说明(请务必先读):
    - 纯 CPU 环境下, 单帧(检测+换脸)通常需要数秒; 因此这里的"实时"特指:
      画面持续跟手、绝不堆积延迟 —— 低帧率准实时, 不是 30fps 视频。
    - 所有优化均围绕「CPU 每帧成本」展开:
        ① 输入帧在进 AI 管道前先用 OpenCV 等比压到 width(默认320) 宽;
        ② 人脸检测采用【跳帧机制】: 每 det_interval 帧才检测一次,
           中间帧直接复用上一检测帧的 bounding box + 关键点(几乎零成本);
        ③ 源脸(提供脸)的特征只在【上传时提取一次】, 后续逐帧复用;
        ④ onnxruntime 显式 CPUExecutionProvider + SessionOptions
           (intra=4 / inter=1 / ORT_ENABLE_ALL) 调优;
        ⑤ 线程安全 + 丢帧保时效: 摄像头每帧一个 HTTP POST 进来,
           「最新帧槽」只留最新一帧, 上一帧没处理完的新帧直接覆盖丢弃,
           由单个后台 worker 顺序消费, 保证线程安全、绝不堆积。
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

# Demo 不上报 Gradio 使用统计
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

BASE = os.path.dirname(os.path.abspath(__file__))
SWAPPER_PATH = os.path.join(BASE, "models", "inswapper_128.onnx")


def build_sess_options(intra: int, inter: int) -> ort.SessionOptions:
    """构造统一调优的 ONNX CPU 会话配置。

    注意: 之前尝试在加载后调用 session.set_intra_op_num_threads 无效
    (insightface 2.1 的 PickableInferenceSession 没有该方法), 且 insightface
    的 FaceAnalysis / get_model 均会透传 sess_options 给 ORT,
    所以线程数必须在【创建会话时】通过 sess_options 指定。
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, intra)   # 每个算子/图内的并行线程
    opts.inter_op_num_threads = max(1, inter)   # 算子间并行(置1避免多图互相抢核)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def make_placeholder(text: str, width: int = 320, height: int = 240) -> np.ndarray:
    """生成一张带提示文字的占位图(RGB), 用于首帧/待命时展示。"""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


class RealTimeSwap:
    """实时换脸状态机: 最新帧槽 + 单 worker 丢帧保时效 + 跳帧检测。

    线程模型:
        - Gradio 的 .stream() 每收到一帧就调用一次 predict_stream()(HTTP 回调线程);
        - predict_stream() 只做两件事: 把新帧覆盖进「最新帧槽」, 然后立刻返回
          「上一帧完成图 + 当前状态」;  无论 CPU 多慢都只花几微秒, 绝不阻塞回调;
        - 后台 worker 线程顺序消费「最新帧槽」, 处理期间回调继续进来的帧只会
          覆盖槽位(即被丢弃), 处理完后 worker 再取最新一帧 —— 这就是丢帧保时效。
    """

    def __init__(self, frame_width: int = 320, det_size: int = 320,
                 det_interval: int = 3, intra_threads: int = 4,
                 inter_threads: int = 1):
        self.width = frame_width            # 输入帧等比例降到的宽度
        self.det_size = self.det = det_size  # 人脸检测分辨率
        self.interval = max(1, det_interval)  # 跳帧: 每 N 帧检测一次
        self._sess_opts = build_sess_options(intra_threads, inter_threads)

        # ---- 跨线程共享状态: 一律用 self._lock 保护 ----
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest = None          # 最新帧槽(可被新帧覆盖 = 丢帧)
        self._result = make_placeholder("WAIT FIRST FRAME")  # 最近一次成品图
        self._status = "模型加载中……"
        self._received = 0           # 回调收到的帧数
        self._processed = 0          # worker 已处理帧数
        self._swapped = 0            # 其中成功换脸帧数
        self._source = None          # 源脸(Face), 只提取一次
        self._target = None          # 缓存的目标脸(Face), 跳帧更新
        self._seq = 0                # 帧序号
        self._last_det_seq = -1      # 最近一次检测所在的帧序号
        self._times = deque(maxlen=10)   # 最近 10 帧耗时(秒), 用于平滑 FPS
        self._fps = 0.0

        # ---- 模型(懒加载, 全程序只加载一次) ----
        self._app = None
        self._swapper = None

        # ---- worker 线程控制 ----
        self._stop = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 模型加载(启动时预热, 避免首帧假死; 首次加载通常需数秒到十几秒)
    # ------------------------------------------------------------------
    def ensure_models(self):
        if self._app is not None:
            return
        if not os.path.isfile(SWAPPER_PATH):
            sys.exit(f"缺少模型 {SWAPPER_PATH}, 请先运行: ./venv/bin/python download_models.py")
        t0 = time.time()
        print("[预热] 加载人脸分析器 buffalo_l ...", flush=True)
        self._app = FaceAnalysis(
            name="buffalo_l", root=BASE,
            providers=["CPUExecutionProvider"], sess_options=self._sess_opts)
        self._app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
        print(f"[预热] 加载换脸器 inswapper_128 ... ({time.time()-t0:.0f}s)", flush=True)
        self._swapper = insightface.model_zoo.get_model(
            SWAPPER_PATH, providers=["CPUExecutionProvider"], sess_options=self._sess_opts)
        # 用一张黑图做一次检测 warmup, 提前分配内存/arena, 避免首帧抖动
        _ = self._app.get(np.zeros((240, 320, 3), dtype=np.uint8))
        print(f"[预热] 模型就绪, 共 {time.time()-t0:.0f}s", flush=True)
        with self._lock:
            self._status = "模型就绪。请上传源脸照片并开启摄像头。"

    def start_worker(self):
        """启动后台 worker 线程(只消费最新帧)。"""
        self._thread = threading.Thread(target=self._workloop, name="swap-worker", daemon=True)
        self._thread.start()

    def stop_worker(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # 源脸(提供脸): 只在用户上传时提取一次, 之后逐帧复用
    # ------------------------------------------------------------------
    def set_source(self, src_rgb):
        """由 src.change 触发: 对上传的源脸照片提取人脸特征(仅一次)。"""
        if self._app is None:
            self.ensure_models()
        if src_rgb is None:
            with self._lock:
                self._source = None
                self._status = "请先上传源脸照片(提供脸)。"
            return self._status
        try:
            bgr = self._resize_bgr(src_rgb)
            faces = self._app.get(bgr)
            if not faces:
                raise ValueError("源图里没检测到人脸, 请换更清楚/居中的照片")
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            with self._lock:
                self._source = face
                self._status = f"源脸已就绪(特征已提取一次): {len(faces)} 张人脸, 取最大。"
            return self._status
        except Exception as e:
            with self._lock:
                self._source = None
                self._status = f"源脸提取失败: {e}"
            return self._status

    # ------------------------------------------------------------------
    # 每帧预处理: 降分辨率(BGR)
    # ------------------------------------------------------------------
    def _resize_bgr(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if w > self.width:                      # 强压到最小可行尺寸(320 宽)
            nw = self.width
            nh = max(1, int(h * nw / w))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Gradio .stream() 回调: 帧进来 -> 覆盖最新帧槽 -> 立刻返回上一帧成品
    # ------------------------------------------------------------------
    def predict_stream(self, frame_rgb):
        """Gradio stream 事件回调 (即方案里的 predict_stream)。

        线程安全: 整个函数在锁内只做"覆盖槽位 + 读出成品", 不跑任何 AI。
        CPU 未处理完时新帧直接把槽位覆盖 = 自动丢帧, 绝不堆积、绝不阻塞回调。
        """
        with self._lock:
            self._received += 1
            if frame_rgb is not None:
                self._latest = frame_rgb        # 丢帧: 覆盖, 而不是排队
            self._cv.notify_all()               # 唤醒 worker
            result = self._result.copy()        # 返回最近一次完成的成品
            status = self._status
        return result, status

    # ------------------------------------------------------------------
    # 后台 worker: 顺序消费最新帧
    # ------------------------------------------------------------------
    def _workloop(self):
        while not self._stop:
            with self._cv:
                while self._latest is None and not self._stop:
                    self._cv.wait(timeout=0.05)
                if self._stop:
                    break
                frame = self._latest
                self._latest = None
            if frame is not None:
                try:
                    self._process_frame(frame)
                except Exception as e:          # 单帧出错不能弄死 worker
                    with self._lock:
                        self._status = f"处理异常: {e}"

    # ------------------------------------------------------------------
    # 单帧主处理: 降分辨率 -> 跳帧检测 -> 换脸 -> 贴状态文字
    # ------------------------------------------------------------------
    def _process_frame(self, frame_rgb: np.ndarray):
        t0 = time.perf_counter()

        # 1) 进 AI 管道前强压分辨率(方案二)
        small_bgr = self._resize_bgr(frame_rgb)

        with self._lock:
            source = self._source
        if source is None:
            display = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            overlay = "UPLOAD SOURCE FIRST"
            swapped_ok = False
        else:
            # 2) 跳帧检测(方案四): 每 interval 帧检测一次, 中间帧复用缓存目标脸
            self._seq += 1
            seq = self._seq
            do_detect = (self._target is None) or (seq - self._last_det_seq >= self.interval)
            target = self._target
            if do_detect:
                faces = self._app.get(small_bgr)
                target = None
                if faces:                       # 换目标图里最大的一张脸
                    target = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                with self._lock:
                    self._target = target
                    self._last_det_seq = seq

            # 3) 换脸(方案三: 纯 CPU inswapper 单图直换)
            if target is None:
                display = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
                overlay = "NO FACE DETECTED"
                swapped_ok = False
            else:
                out = self._swapper.get(small_bgr, target, source, paste_back=True)
                display = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                overlay = "LIVE SWAP | DROP-FRAME ON"
                swapped_ok = True

        # 4) 帧耗时 / FPS 平滑统计
        dt = time.perf_counter() - t0
        with self._lock:
            self._processed += 1
            if swapped_ok:
                self._swapped += 1
            self._times.append(dt)
            # 样本未满时也用已有样本估 FPS, 让首帧后立即有实时帧率可看
            self._fps = 1.0 / (sum(self._times) / len(self._times))
            ms = dt * 1000
            dropped = self._received - self._processed      # 丢帧数 = 收-处理
            self._status = (
                f"纯CPU实时换脸·丢帧保时效 | 本帧 {ms:.0f}ms | 平滑 ~{self._fps:.2f} FPS | "
                f"已处理 {self._processed} / 成功换脸 {self._swapped} / 丢帧 {dropped} | "
                f"检测每 {self.interval} 帧一次(det={self.det_size})"
            )

        # 5) 在预览上贴英文叠加信息(中文由状态栏/textbox 显示, cv2 不支持中文)
        cv2.putText(display, overlay, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"{ms:.0f} ms/frame", (8, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)

        with self._lock:
            self._result = display.copy()
        return self._result


# =====================================================================
# Gradio 界面
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="纯 CPU 实时换脸(丢帧保时效) Web 界面")
    ap.add_argument("--port", type=int, default=7861, help="监听端口(默认 7861)")
    ap.add_argument("--width", type=int, default=320, help="输入帧降到的宽度(默认320)")
    ap.add_argument("--det-size", type=int, default=320, help="人脸检测分辨率(默认320, 越小越快)")
    ap.add_argument("--det-interval", type=int, default=3, help="跳帧: 每 N 帧检测一次(默认3)")
    ap.add_argument("--intra-threads", type=int, default=4, help="ONNX intra 线程(默认4)")
    ap.add_argument("--inter-threads", type=int, default=1, help="ONNX inter 线程(默认1)")
    args = ap.parse_args()

    state = RealTimeSwap(frame_width=args.width, det_size=args.det_size,
                         det_interval=args.det_interval,
                         intra_threads=args.intra_threads,
                         inter_threads=args.inter_threads)
    state.ensure_models()       # 启动时预热, 避免首帧假死
    state.start_worker()        # 启动后台消费线程

    import gradio as gr

    def change_det(v):
        state.det_size = state.det = int(v)
        state._app.prepare(ctx_id=0, det_size=(v, v))
        return f"检测分辨率已设为 {v}"

    def change_interval(v):
        state.interval = max(1, int(v))
        return f"跳帧检测间隔已设为 {state.interval} 帧"

    with gr.Blocks(title="纯CPU实时换脸 (丢帧保时效)") as demo:
        gr.Markdown(
            "## 🎥 纯 CPU 实时换脸测试 — **已开启丢帧保时效策略**\n"
            "- 摄像头每帧推送到后台, 后台只处理**最新一帧**, 未处理完的帧直接丢(宁可降帧率绝不堆积延迟)\n"
            "- 人脸检测**每 N 帧跳帧一次**, 中间帧复用边界框/关键点; 源脸特征**只提取一次**\n"
            "- 输入帧强压到 320 宽再进 AI 管道, ONNX 显式 CPU 会话(线程已调优)\n"
            "- ⚠️ 纯 CPU 下每帧通常需数秒, 帧率会明显偏低 —— **低帧率准实时(丢帧保时效), 不是 30fps 视频**; 有 GPU 时把 providers 加上 CUDAExecutionProvider 即可显著提速"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ① 源脸(提供脸)")
                src = gr.Image(label="上传一次即可, 特征只提一次",
                               type="numpy", sources=["upload"], height=300)
                det = gr.Radio([192, 320, 640], value=args.det_size,
                               label="人脸检测分辨率(越小越快)")
                interval = gr.Slider(1, 10, value=args.det_interval, step=1,
                                     label="检测间隔帧数(跳帧)")
            with gr.Column(scale=1):
                gr.Markdown("### ② 摄像头(被换脸)")
                cam = gr.Image(label="开启摄像头后自动推流",
                               sources=["webcam"], streaming=True,
                               type="numpy", height=300)
        with gr.Row():
            preview = gr.Image(label="③ 实时换脸预览(丢帧保护生效)",
                               type="numpy", interactive=False, streaming=True, height=300)
        status = gr.Textbox(label="状态", lines=3)

        # webcam 流: 每帧回调 -> 只写槽 -> 立刻返回上一帧成品
        cam.stream(fn=state.predict_stream, inputs=[cam], outputs=[preview, status],
                   stream_every=0.10, concurrency_limit=1)
        # 源脸上传 / 参数调整
        src.change(fn=state.set_source, inputs=[src], outputs=[status])
        det.change(fn=change_det, inputs=[det], outputs=[status])
        interval.change(fn=change_interval, inputs=[interval], outputs=[status])

    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port,
        show_error=True, quiet=True,
    )
    state.stop_worker()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止")
