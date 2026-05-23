"""
一键启动 TextPatch Studio 前端
Usage: python start.py
"""
import subprocess
import webbrowser
import time
import sys
import os
from pathlib import Path

PORT = 8000
HOST = "127.0.0.1"
URL = f"http://localhost:{PORT}"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def kill_port(port):
    """Kill any process occupying the given port on Windows."""
    result = subprocess.run(
        ["netstat", "-ano"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            parts = line.strip().split()
            pid = parts[-1]
            print(f"  端口 {port} 被 PID {pid} 占用，正在终止...")
            subprocess.run(["cmd", "/c", f"taskkill /F /PID {pid}"],
                           capture_output=True)
            time.sleep(2)
            return True
    print("  端口空闲")
    return False


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TextPatch Studio 一键启动")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--kill-port", action="store_true", help="显式终止占用端口的进程")
    return parser.parse_args()


def _has_uvicorn(python_path):
    """Return True if the Python at python_path has uvicorn installed."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import uvicorn; print(uvicorn.__file__)"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_python():
    """Find a Python interpreter that has uvicorn installed.

    Checks, in order:
      1. sys.executable (the Python that launched this script)
      2. A sibling conda env next to the project dir (e.g. .../gptimage2.conda/python.exe)
      3. python / python3 on PATH
    """
    # 1. The interpreter that is running this script
    if _has_uvicorn(sys.executable):
        return sys.executable

    # 2. Project-local conda env — sibling to project directory
    parent = Path(PROJECT_DIR).parent
    for conda_name in ["gptimage2.conda", ".conda", "conda_env", "env"]:
        candidate = parent / conda_name / "python.exe"
        if candidate.exists() and _has_uvicorn(str(candidate)):
            return str(candidate)

    # 3. Generic python / python3 on PATH
    for cmd in ["python", "python3"]:
        if _has_uvicorn(cmd):
            return cmd

    return None


def main():
    args = _parse_args()
    global HOST, PORT
    HOST = args.host
    PORT = args.port

    os.chdir(PROJECT_DIR)
    print(f"TextPatch Studio — 一键启动")
    print(f"项目目录: {PROJECT_DIR}")

    # 1. Find a working Python with uvicorn
    python_exe = find_python()
    if not python_exe:
        print("\n  错误: 未找到安装有 uvicorn 的 Python 环境。")
        print("  请先安装: pip install uvicorn")
        print("  或使用 Anaconda: conda install uvicorn")
        sys.exit(1)
    print(f"Python:    {python_exe}\n")

    # 2. Check port
    print("[1/3] 检查端口占用...")
    if args.kill_port:
        kill_port(PORT)
    else:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        port_in_use = any(
            f":{PORT}" in line and "LISTENING" in line
            for line in result.stdout.splitlines()
        )
        if port_in_use:
            print(f"  端口 {PORT} 已被占用。可使用 --port {PORT+1} 或 --kill-port 显式终止。")
            sys.exit(1)
        print("  端口空闲")

    # 3. Start uvicorn server
    print(f"[2/3] 启动后端服务...")
    server = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app",
         "--host", HOST, "--port", str(PORT)],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # 4. Wait for server to be ready
    print("[3/3] 等待服务就绪...")
    import urllib.request
    import threading

    output_lines = []

    def read_output():
        for line in server.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                if any(kw in line for kw in ["ERROR", "error", "Traceback", "ModuleNotFound"]):
                    print(f"  [SERVER] {line}")

    t = threading.Thread(target=read_output, daemon=True)
    t.start()

    # Give the process a moment to fail fast
    time.sleep(1)
    if server.poll() is not None:
        time.sleep(0.5)  # Let thread collect output
        print(f"\n  服务器启动失败! (exit code: {server.returncode})")
        if output_lines:
            print("  服务端输出:")
            for line in output_lines[-15:]:
                print(f"    {line}")
        print(f"\n  提示: 请确认 {python_exe} 中已安装: pip install uvicorn")
        sys.exit(1)

    started = False
    for i in range(30):
        if server.poll() is not None:
            time.sleep(0.5)
            print(f"\n  服务器意外退出! (exit code: {server.returncode})")
            for line in output_lines[-15:]:
                print(f"    {line}")
            sys.exit(1)
        try:
            resp = urllib.request.urlopen(URL, timeout=1)
            if resp.status == 200:
                started = True
                break
        except Exception:
            time.sleep(0.5)
    else:
        if server.poll() is not None:
            print(f"\n  服务器进程已退出 (exit code: {server.returncode})")
            for line in output_lines[-10:]:
                print(f"    {line}")
            sys.exit(1)

    print(f"\n  前端已启动: {URL}")
    webbrowser.open(URL)
    print("\n按 Ctrl+C 停止服务...")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        print("已停止。")


if __name__ == "__main__":
    main()
