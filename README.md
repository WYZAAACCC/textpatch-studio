# TextPatch Studio

AI 生成图中文字后处理工具 — 检测乱码小字、OCR/LLM 校正、擦除伪字、真实字体重新排版。

## 解决的问题

GPT Image、Midjourney、Stable Diffusion、Flux 等图像生成模型在生成小字、远景文字、中文标牌、海报说明时，常见文字模糊、伪汉字、字形错误、多字混叠等问题。

TextPatch Studio 提供三步后处理流程：

1. **检测** — OCR 识别图片中的乱码/伪文字区域
2. **擦除** — Inpainting 擦掉原来的伪文字
3. **回填** — 用真实字体和确定性排版将正确文字渲染回去

## 项目结构

```
textpatch-studio/
├── backend/             # FastAPI Python 后端
│   ├── api/             # REST API 路由
│   ├── core/            # 核心流程 (Pipeline)
│   ├── models/          # 数据模型
│   ├── llm_adapters/    # LLM 适配器 (DeepSeek)
│   ├── ocr_adapters/    # OCR 适配器 (RapidOCR / PaddleOCR)
│   ├── inpaint_adapters/# Inpainting 适配器 (OpenCV)
│   ├── renderer/        # 文字渲染引擎
│   └── storage/         # 存储层
├── frontend/            # Vite + Konva.js 前端
│   ├── js/              # 核心 JS 模块
│   ├── css/             # 样式
│   └── src/             # React 组件 (TypeScript)
├── cli/                 # 命令行工具
├── scripts/             # 工具脚本
├── tests/               # 测试
├── fonts/               # 字体目录 (运行时下载)
└── data/                # 项目数据 (运行时生成, gitignored)
```

## 快速开始

### 环境要求

- Python >= 3.11
- Node.js >= 18 (前端开发)

### 安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 可选: 安装本地 OCR 引擎
pip install rapidocr-onnxruntime

# 前端依赖 (开发模式)
cd frontend && npm install
```

### 配置

```bash
# 方式一: 环境变量
export DEEPSEEK_API_KEY=sk-your-key

# 方式二: 在项目根目录创建 apikey.txt
echo "sk-your-key" > apikey.txt
```

详见 `.env.example` 了解所有配置项。

### 运行

```bash
# 一键启动 (自动查找 Python 环境并启动后端 + 打开前端)
python start.py

# 或手动启动
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 打开前端界面。

### CLI 模式

```bash
# 直接处理单张图片
python run_pipeline.py "path/to/image.png"

# 批量处理
python run_batch.py
```

## API 概览

| 路径 | 说明 |
|------|------|
| `GET /api/health` | 系统健康检查 |
| `POST /api/projects` | 创建项目并上传图片 |
| `POST /api/regions/detect` | 检测文字区域 |
| `POST /api/ocr/recognize` | OCR 识别 |
| `POST /api/correction/correct` | LLM 文字校正 |
| `POST /api/inpaint/erase` | 擦除伪文字 |
| `POST /api/render/render` | 渲染真实文字 |
| `POST /api/export/export` | 导出最终图片 |
| `GET /api/fonts` | 列出可用字体 |
| `POST /api/fonts/download` | 下载默认中文字体 |

## 技术栈

- **后端**: FastAPI + Uvicorn + OpenCV + Pillow + scikit-image
- **前端**: Vite + Konva.js + React + Zustand
- **OCR**: RapidOCR (本地) / PaddleOCR (API)
- **LLM**: DeepSeek API
- **Inpainting**: OpenCV (Telea / NS)

## License

MIT License — 详见 [LICENSE](LICENSE)
