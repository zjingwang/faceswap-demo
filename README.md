# faceswap-demo

基于本地 ONNX 的 **AMD DirectML 换脸** Web 平台。使用 insightface `buffalo_l`（检测/关键点/512-d embedding）+ `inswapper_128.onnx`，面向 **Ryzen + Radeon 780M（共享内存、无独显 VRAM、无 NPU）** 优化。

> 推理后端固定为 `['DmlExecutionProvider', 'CPUExecutionProvider']`，**不使用 CUDA**。全局 `threading.Lock`（`ai_core.INFERENCE_LOCK`）串行化 AI。

## 功能

| Tab | 能力 |
| --- | --- |
| ① Webcam 实时换脸 | 摄像头推流 + 上传源脸；AI 前压到 320 宽再拉回；源脸 embedding 只提一次；上一帧未完成则丢帧 |
| ② 离线视频换脸 | 源脸 + MP4；逐帧 `VideoCapture`/`VideoWriter`；尽量保留原音轨 |
| ③ 究极融合（脸+声） | 双栏 UI；全员鬼畜 / 精准人脸编号；ffmpeg 抽音 + 源声音循环填充 + mux |

另提供命令行 `faceswap.py` 与独立入口 `webcam_realtime.py`。

## 目录结构

```
faceswap-demo/
├── app.py / web_app.py  # 主 Gradio（3 Tabs）
├── ai_core.py           # DirectML 模型加载、全局推理锁、换脸工具
├── webcam_session.py    # 摄像头丢帧会话（embedding 缓存）
├── voice_utils.py       # ffmpeg 抽音 / 人声对齐 / mux
├── webcam_realtime.py   # 仅摄像头入口（可选）
├── faceswap.py          # 命令行照片换脸
├── download_models.py   # 下载 buffalo_l + inswapper_128.onnx
├── requirements.txt     # onnxruntime-directml（无 CUDA）
├── models/              # 模型目录（不入库）
│   ├── buffalo_l/
│   └── inswapper_128.onnx
├── input/
└── output/
```

## 环境准备（Windows + AMD）

需要 **Python 3.10–3.12**（推荐 3.12；3.13+ 可能缺少 DirectML 轮子）、**ffmpeg**（加入 PATH，Tab②/③ 人声封装需要）。

```powershell
git clone https://github.com/zjingwang/faceswap-demo.git
cd faceswap-demo

# 建议用 Python 3.12 建 venv
py -3.12 -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# insightface 可能顺带装上 CPU 版 onnxruntime，会盖掉 DirectML —— 必须卸掉：
pip uninstall -y onnxruntime

python download_models.py

winget install Gyan.FFmpeg   # 若尚未安装 ffmpeg；新开终端后再测 ffmpeg -version
```

### 模型放置路径

| 文件 | 路径 |
| --- | --- |
| 换脸模型 | `models/inswapper_128.onnx`（~550MB） |
| 人脸分析包 | `models/buffalo_l/*.onnx`（~340MB） |

手动下载（insightface v0.7）：

- https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx
- https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip → 解压到 `models/buffalo_l/`

> FaceAnalysis 的 `root` 为项目根目录，buffalo_l 必须在 `<项目>/models/buffalo_l/`。

### DirectML 说明

- 依赖包是 **`onnxruntime-directml`**，不要同时装 `onnxruntime` / `onnxruntime-gpu`。
- providers 显式为：`['DmlExecutionProvider', 'CPUExecutionProvider']`。
- 780M 使用共享系统内存；视频结束后 `cap.release()` + `gc.collect()`。
- 无 CUDA / 无 NPU 依赖。

## 运行

```powershell
.\.venv\Scripts\activate
python app.py
# 或 python web_app.py
# 浏览器打开 http://127.0.0.1:7860
```

### Tab1：摄像头实时换脸

1. 上传源脸（embedding 只提取一次）
2. 开启摄像头
3. 帧压到 320 宽进 AI，再拉回原尺寸；AI 忙时丢弃/覆盖当前帧，不堆积

### Tab2：离线视频换脸 + 人声（双栏）

**左栏**：① 源脸照片 ② 源声音片段 ③ 目标视频 ·「全员鬼畜模式」·「精准目标人脸编号」· 启动按钮  
**右栏**：① 最终视频 ② 中文日志

- 全员鬼畜 = True：替换所有人脸  
- = False：第 1 帧按 x 从左到右排序，只换指定编号  
- 视频 AI 宽默认 640（再拉回原分辨率）；进度条 0–100%

人声（本地可跑，非完整 RVC）：

1. ffmpeg 从目标视频抽音  
2. 若有源声音：循环/填充到视频时长后替换  
3. 否则保留原音轨  
4. mux → `output/fusion_final_*.mp4`

> **限制**：不是 RVC/So-VITS 级音色克隆，而是 ffmpeg 最佳努力替换。可自行改 `voice_utils.process_voice`。

### 命令行

```powershell
python faceswap.py --source input/biden.jpg --target input/obama.jpg --output output/demo.png
```

## 常见问题

- 缺少模型：先跑 `python download_models.py`  
- 找不到 ffmpeg：`winget install Gyan.FFmpeg` 后确认 `ffmpeg -version`  
- DirectML 回退 CPU / `get_available_providers` 异常：`pip uninstall -y onnxruntime` 后确认 `pip show onnxruntime-directml`，再 `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` 应含 `DmlExecutionProvider`
- `download_models.py` 解压后应出现 `models/buffalo_l/*.onnx`（官方 zip 可能是扁平布局，脚本会归位）
- 不要开 GFPGAN/CodeFormer：共享显存机器上极易卡死  


## License

MIT
