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
├── backend/             # FastAPI Python 后端
│   ├── api/             # REST API 路由
│   ├── core/            # 核心流程 (Pipeline)
│   ├── models/          # 数据模型
│   ├── llm_adapters/    # LLM 适配器 (DeepSeek / Mock)
│   ├── ocr_adapters/    # OCR 适配器 (RapidOCR / PaddleOCR)
│   ├── inpaint_adapters/# Inpainting 适配器 (OpenCV Telea/NS + SimpleFill)
│   ├── renderer/        # 文字渲染引擎 (PIL)
│   └── storage/         # 存储层
├── frontend/            # Vanilla JS + Fabric.js 前端 (Vite 静态服务)
│   ├── js/              # 核心 JS 模块
│   ├── css/             # 样式
│   └── index.html       # 主页面
├── cli/                 # 命令行工具
├── tests/               # 测试
├── fonts/               # 字体目录 (手动放入或开启下载)
└── data/                # 项目数据 (运行时生成)
```

## 快速开始

### 环境要求

- Python >= 3.11

### 安装

```bash
pip install -r requirements.txt

# 可选: 安装本地 OCR 引擎
pip install rapidocr-onnxruntime

# 可选: 安装开发依赖
pip install -e ".[dev]"
```

### 配置

```bash
# 环境变量 (推荐)
export DEEPSEEK_API_KEY=sk-your-key

# 或: 在项目根目录创建 apikey.txt
echo "sk-your-key" > apikey.txt
```

详见 [.env.example](.env.example) 了解所有配置项。

### 运行

```bash
# 一键启动 (默认监听 127.0.0.1:8000)
python start.py

# 手动启动
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 公网部署时需开启认证
TEXTPATCH_REQUIRE_AUTH=true TEXTPATCH_API_TOKEN=your-token python start.py --host 0.0.0.0
```

访问 `http://localhost:8000` 打开前端界面。

### CLI 模式

```bash
# 安装 CLI
pip install -e .

# 命令行使用
textpatch --help
textpatch serve --host 127.0.0.1 --port 8000
```

## 安全说明

- 默认监听 `127.0.0.1`（仅本地访问）
- 公网部署请设置 `TEXTPATCH_REQUIRE_AUTH=true` 和 `TEXTPATCH_API_TOKEN`
- 远程字体下载默认关闭，需 `TEXTPATCH_ENABLE_FONT_DOWNLOAD=true` 开启
- 远程 LLM/OCR 调用需显式配置 API key
- 无 API key 时 Mock LLM 会明确标记 `confidence=0, needs_human=true`

详见 [.env.example](.env.example) 的安全配置项。

## API 概览

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 系统健康检查 (公开) |
| `/api/projects` | POST | 创建项目并上传图片 |
| `/api/projects` | GET | 列出所有项目 |
| `/api/projects/{id}` | GET | 获取项目详情 |
| `/api/projects/{id}/regions` | GET | 列出项目文字区域 |
| `/api/projects/{id}/regions/{region_id}` | PATCH | 更新区域文本/样式/状态 |
| `/api/projects/{id}/detect` | POST | 检测文字区域 |
| `/api/projects/{id}/ocr` | POST | OCR 识别 |
| `/api/projects/{id}/correct` | POST | LLM 文字校正 (SSE) |
| `/api/projects/{id}/inpaint` | POST | 擦除伪文字 |
| `/api/projects/{id}/render` | POST | 渲染真实文字 |
| `/api/projects/{id}/export` | POST | 导出 (png/jpeg/webp/zip) |
| `/api/projects/{id}/image/{type}` | GET | 获取项目图片 |
| `/api/projects/{id}/restore` | POST | 还原区域背景 |
| `/api/fonts` | GET | 列出可用字体 (公开) |
| `/api/fonts/download` | POST | 下载默认中文字体 |

## 技术栈

- **后端**: FastAPI + Uvicorn + OpenCV + Pillow + scikit-image
- **前端**: Vanilla JS + Fabric.js + Vite
- **OCR**: RapidOCR (本地) / PaddleOCR (API)
- **LLM**: DeepSeek API
- **Inpainting**: OpenCV (Telea / NS) + SimpleFill

## License

MIT License — 详见 [LICENSE](LICENSE)
