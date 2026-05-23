from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="textpatch",
        description="TextPatch Studio - AI生成图中文字后处理工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    repair_parser = subparsers.add_parser("repair", help="一键修复图片文字")
    repair_parser.add_argument("input", help="输入图片路径")
    repair_parser.add_argument("--output", "-o", help="输出图片路径")
    repair_parser.add_argument("--lang", default="zh-CN", help="语言")
    repair_parser.add_argument("--llm", default="deepseek-v4-flash", help="LLM模型名")
    repair_parser.add_argument("--font", help="字体文件路径")
    repair_parser.add_argument("--auto-accept", action="store_true", help="自动接受校正")

    detect_parser = subparsers.add_parser("detect", help="只检测文字区域")
    detect_parser.add_argument("input", help="输入图片路径")
    detect_parser.add_argument("--output", "-o", help="输出JSON路径")

    erase_parser = subparsers.add_parser("erase", help="只擦除文字")
    erase_parser.add_argument("input", help="输入图片路径")
    erase_parser.add_argument("--regions", help="区域JSON路径")
    erase_parser.add_argument("--output", "-o", help="输出图片路径")

    render_parser = subparsers.add_parser("render", help="按项目回填文字")
    render_parser.add_argument("--project", help="项目目录路径")
    render_parser.add_argument("--output", "-o", help="输出图片路径")

    serve_parser = subparsers.add_parser("serve", help="启动API服务")
    serve_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口")

    args = parser.parse_args()

    if args.command == "repair":
        _cmd_repair(args)
    elif args.command == "detect":
        _cmd_detect(args)
    elif args.command == "erase":
        _cmd_erase(args)
    elif args.command == "render":
        _cmd_render(args)
    elif args.command == "serve":
        _cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_repair(args):
    from backend.core.pipeline import Pipeline
    from backend.config import app_config

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    if args.llm:
        app_config.llm.model = args.llm

    pipeline = Pipeline(app_config)

    print(f"创建项目: {input_path.name}")
    project = pipeline.create_project(input_path.stem, input_path)
    print(f"项目ID: {project.id}")

    print("检测文字区域...")
    project = pipeline.detect(project.id)
    print(f"检测到 {len(project.regions)} 个文字区域")

    print("OCR识别...")
    project = pipeline.ocr(project.id)

    print("LLM校正...")
    project = pipeline.correct(project.id, auto_accept=args.auto_accept)

    needs_review = [r for r in project.regions if r.status == "needs_review"]
    if needs_review:
        print(f"⚠ {len(needs_review)} 个区域需要人工审核")

    print("擦除文字...")
    project = pipeline.inpaint(project.id)

    print("渲染文字...")
    project = pipeline.render(project.id)

    print("导出...")
    output_path = pipeline.export_project(project.id)
    if args.output:
        import shutil
        shutil.copy2(str(output_path), args.output)
        output_path = Path(args.output)

    print(f"✓ 完成! 输出: {output_path}")


def _cmd_detect(args):
    from backend.core.pipeline import Pipeline
    from backend.config import app_config

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    pipeline = Pipeline(app_config)
    project = pipeline.create_project(input_path.stem, input_path)
    project = pipeline.detect(project.id)

    output = {
        "project_id": project.id,
        "regions": [r.to_dict() for r in project.regions],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"检测结果已保存到: {args.output}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _cmd_erase(args):
    import cv2
    from backend.core.pipeline import Pipeline
    from backend.config import app_config

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    pipeline = Pipeline(app_config)

    if args.regions:
        with open(args.regions, "r", encoding="utf-8") as f:
            regions_data = json.load(f)
        from backend.models.region import TextRegion
        regions = [TextRegion.from_dict(r) for r in regions_data.get("regions", [])]
    else:
        project = pipeline.create_project(input_path.stem, input_path)
        project = pipeline.detect(project.id)
        regions = project.regions

    image = cv2.imread(str(input_path))
    from backend.core.inpainting import inpaint_regions
    clean_base, _ = inpaint_regions(image, regions)

    output_path = args.output or str(input_path.with_stem(input_path.stem + "_clean"))
    cv2.imwrite(output_path, clean_base)
    print(f"擦除结果已保存到: {output_path}")


def _cmd_render(args):
    from backend.core.pipeline import Pipeline
    from backend.config import app_config

    pipeline = Pipeline(app_config)

    if not args.project:
        print("错误: 请指定项目目录 --project")
        sys.exit(1)

    project_dir = Path(args.project)
    json_path = project_dir / "project.json"
    if not json_path.exists():
        print(f"错误: 项目文件不存在: {json_path}")
        sys.exit(1)

    from backend.storage.project_store import ProjectStore
    store = ProjectStore(app_config.storage.data_dir)
    project = store.load(project_dir.name.replace(".textpatch", ""))

    if not project:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from backend.models.project import Project
        project = Project.from_dict(data)

    project = pipeline.render(project.id)
    output_path = pipeline.export_project(project.id)

    if args.output:
        import shutil
        shutil.copy2(str(output_path), args.output)
        output_path = Path(args.output)

    print(f"渲染结果已保存到: {output_path}")


def _cmd_serve(args):
    import uvicorn
    from backend.config import app_config

    app_config.host = args.host
    app_config.port = args.port

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=app_config.debug,
    )


if __name__ == "__main__":
    main()
