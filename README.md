# faceswap-demo

基于本地 ONNX 模型的**纯 CPU 照片换脸** Demo。使用 insightface 的 `buffalo_l` 人脸分析模型 + `inswapper_128` 换脸模型，提供命令行和 Web（Gradio）两种使用方式。

> 所有推理都在本机完成（`CPUExecutionProvider`），**不需要 GPU**，图片不会上传到任何服务器。

## 功能特性

- 🧠 纯 CPU 推理，无显卡/虚拟机也能跑，有 GPU 同样可用
- 🌐 Gradio Web 界面，浏览器可视化操作
- ⌨️ 命令行脚本，适合批量/脚本化调用
- 📦 模型通过脚本一键下载，示例图片随仓库提供

## 目录结构

```
faceswap-demo/
├── faceswap.py          # 命令行换脸脚本
├── web_app.py           # Gradio Web 界面（纯 CPU）
├── download_models.py   # 下载模型脚本（buffalo_l + inswapper_128.onnx）
├── requirements.txt     # Python 依赖
├── models/              # 模型目录（不入库，运行 download_models.py 生成）
│   ├── buffalo_l/       #   insightface 人脸分析模型包（约 340MB）
│   └── inswapper_128.onnx   #   换脸模型（约 550MB）
├── input/               # 示例图片（源脸 / 目标）
├── output/              # 换脸结果示例（运行后自动生成）
├── venv/                # 本地 Python 虚拟环境（不入库）
└── README.md
```

## 环境准备

需要 **Python 3.10+**（作者用 Python 3.10 验证通过）。

```bash
# 1. 克隆仓库
git clone https://github.com/zjingwang/faceswap-demo.git
cd faceswap-demo

# 2. 创建虚拟环境（推荐，避免污染系统 Python）
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. 下载模型（合计约 0.9GB，请保持网络畅通；可断点续传，重复执行会自动跳过已下载文件）
python download_models.py
```

> 说明：`venv/` 和 `models/` 都没有入库。前者是本地环境不应上传；后者因单文件超过 GitHub 100MB 限制由脚本按需下载，克隆仓库后执行上一步即可得到完整模型。

## 运行 Web 界面

```bash
# 在项目根目录（已激活 venv）执行
python web_app.py            # 默认端口 7860
python web_app.py --port 7890   # 使用其他端口
```

然后浏览器打开 **http://127.0.0.1:7860**：

1. **① 源脸照片**：提供人脸的照片
2. **② 目标照片**：被换脸的照片
3. 点击 **🔄 开始换脸**
4. 结果与「源 | 目标 | 结果」对比图同时保存到项目 `output/` 目录

## 命令行使用

```bash
python faceswap.py \
    --source input/biden.jpg \
    --target input/obama.jpg \
    --output output/biden_obama.png
```

参数说明（`python faceswap.py -h`）：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--source` | 人脸来源图（提供脸） | 必填 |
| `--target` | 目标图（被换脸） | 必填 |
| `--output` | 输出图片路径 | `./output.png` |
| `--models` | 存放 `inswapper_128.onnx` 的目录 | `./models` |
| `--root` | FaceAnalysis 根目录（`buffalo_l` 位于 `<root>/models/buffalo_l`） | 项目根目录 |
| `--det-size` | 人脸检测分辨率，越小越快（建议 320–640） | `640` |

## 模型来源

| 模型 | 大小 | 来源 |
| --- | --- | --- |
| `buffalo_l` 模型包 | ~340MB | insightface 官方 release（`deepinsight/insightface` v0.7） |
| `inswapper_128.onnx` | ~550MB | insightface 官方 release（`deepinsight/insightface` v0.7） |

## 常见问题

- **首次换脸较慢**：模型只会在第一次使用时加载（约 10 秒），之后每次约几秒。
- **检测不到人脸**：戴口罩、侧面、过暗的人脸检测不到，属预期行为，请换更清晰/更居中的照片。多张人脸时，源图取第 1 张，目标图换面积最大的一张。
- **不要开启「面部修复」**（GFPGAN/CodeFormer 类）：那是给有 GPU 的机器用的，纯 CPU 上开启会非常慢甚至卡死。
- **Web 界面只监听 `127.0.0.1`**：仅本机可访问，不会对外网开放。

## License

MIT
