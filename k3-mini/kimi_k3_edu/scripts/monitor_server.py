#!/usr/bin/env python3
"""
monitor_server.py - Live training dashboard server for K3-Edu.

Like https://mimo.xiaomi.com/rl/, it shows a live view of an in-progress
training run: dataset info, loss/perplexity curves, throughput, LR schedule,
cost, ETA, GPU/memory gauges, and checkpoints.

Usage:
    python scripts/monitor_server.py                          # newest run in runs/
    python scripts/monitor_server.py --run runs/run_20260922_101500
    python scripts/monitor_server.py --port 8000

Then open http://localhost:8000 in your browser.

API:
    GET /                      -> web/dashboard.html
    GET /api/runs              -> list of available runs
    GET /api/metrics           -> all metric lines (JSON array), tail-friendly
    GET /api/metrics?after=N   -> lines after byte offset N (for polling)
    GET /api/summary           -> latest snapshot + run metadata
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # kimi_k3_edu/
RUNS_DIR = os.path.join(ROOT, "runs")
WEB_DIR = os.path.join(ROOT, "web")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import detect_hardware as hwmod

# Static hardware info is detected ONCE per server start and cached both in
# memory and to runs/hw_static.json, so the dashboard shows the real machine
# the trainer is running on (device, CPU, RAM, CUDA...) instead of guesses.
_hw_static = None
_hw_detect_err = None

def hw_static_cached(force=False):
    global _hw_static, _hw_detect_err
    if _hw_static is not None and not force:
        return _hw_static
    cache = os.path.join(RUNS_DIR, "hw_static.json")
    if not force and os.path.exists(cache):
        try:
            with open(cache) as f:
                _hw_static = json.load(f)
                return _hw_static
        except (OSError, json.JSONDecodeError):
            pass
    try:
        _hw_static = hwmod.static_info()
    except Exception as e:
        _hw_detect_err = str(e)
        _hw_static = {}
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(_hw_static, f, indent=2)
    except OSError:
        pass
    return _hw_static

_lock = threading.Lock()


def find_run_dir(requested: str | None) -> str | None:
    """Resolve the run directory: explicit arg, or newest under runs/."""
    if requested:
        return requested if os.path.isdir(requested) else None
    if not os.path.isdir(RUNS_DIR):
        return None
    runs = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*")),
                  key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None


def read_metrics(run_dir: str) -> list[dict]:
    """Parse all complete JSON lines from metrics.jsonl."""
    path = os.path.join(run_dir, "metrics.jsonl")
    lines = []
    if not os.path.exists(path):
        return lines
    with _lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # partial line (writer mid-flush) - skip
        except OSError:
            pass
    return lines


def read_bytes_from(path: str, after: int) -> tuple[bytes, int]:
    """Return file bytes after offset `after` (for efficient polling)."""
    if not os.path.exists(path):
        return b"", after
    with _lock:
        try:
            size = os.path.getsize(path)
            if size <= after:
                return b"", after
            with open(path, "rb") as f:
                f.seek(after)
                data = f.read()
            return data, size
        except OSError:
            return b"", after


class Handler(BaseHTTPRequestHandler):
    run_dir: str | None = None

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        if route == "/" or route == "/index.html":
            return self._serve_file("dashboard.html")
        if route.endswith(".html") or route.endswith(".js") or route.endswith(".css"):
            return self._serve_file(route.lstrip("/"))

        if route == "/api/runs":
            runs = []
            if os.path.isdir(RUNS_DIR):
                for d in sorted(glob.glob(os.path.join(RUNS_DIR, "run_*")),
                                key=os.path.getmtime, reverse=True):
                    runs.append({"dir": d, "name": os.path.basename(d),
                                 "modified": os.path.getmtime(d)})
            return self._send_json({"runs": runs, "current": self.run_dir})

        if route == "/api/metrics":
            if not self.run_dir:
                return self._send_json({"error": "no run found under runs/"}, 404)
            after = int(qs.get("after", ["0"])[0])
            if after > 0:
                data, new_off = read_bytes_from(
                    os.path.join(self.run_dir, "metrics.jsonl"), after)
                new_lines = []
                for ln in data.decode("utf-8", errors="replace").splitlines():
                    ln = ln.strip()
                    if ln:
                        try:
                            new_lines.append(json.loads(ln))
                        except json.JSONDecodeError:
                            pass
                return self._send_json({"lines": new_lines, "offset": new_off})
            return self._send_json({"lines": read_metrics(self.run_dir)})

        if route == "/api/hw":
            """Real hardware: static info (detected once) + fresh dynamic sample."""
            static = hw_static_cached()
            try:
                dyn = hwmod.sample_dynamic()
            except Exception as e:
                dyn = {"error": str(e)}
            return self._send_json({"static": static, "dynamic": dyn})

        if route == "/api/summary":
            if not self.run_dir:
                return self._send_json({"error": "no run found"}, 404)
            lines = read_metrics(self.run_dir)
            meta = next((l for l in lines if l.get("event") == "meta"), None)
            steps = [l for l in lines if "step" in l]
            latest = steps[-1] if steps else None
            return self._send_json({
                "run_dir": self.run_dir,
                "meta": meta,
                "latest": latest,
                "n_steps": len(steps),
                "finished": any(l.get("event") in ("finished", "aborted") for l in lines),
            })

        return self._send_json({"error": "not found"}, 404)

    def _serve_file(self, rel):
        path = os.path.join(WEB_DIR, rel)
        if not os.path.isfile(path):
            return self._send_json({"error": f"{rel} not found"}, 404)
        ctype = {"html": "text/html", "js": "application/javascript",
                 "css": "text/css"}.get(rel.rsplit(".", 1)[-1], "text/plain")
        with open(path, "rb") as f:
            return self._send(200, f.read(), ctype)


def acquire_lock(port: int) -> int | None:
    """Single-instance guard.

    Writes a lock file (runs/.monitor.lock) containing our PID + port and
    checks it before binding:
      - if the lock holder is still alive and serving, exit with a message
        pointing at the existing dashboard URL;
      - if the lock holder is dead (crashed / terminal closed), the stale lock
        is removed automatically.
    Returns the PID of the existing server if one is running, else None
    (meaning: we own the lock and may start).
    """
    lock_path = os.path.join(RUNS_DIR, ".monitor.lock")
    os.makedirs(RUNS_DIR, exist_ok=True)

    if os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                other = json.load(f)
            other_pid, other_port = int(other["pid"]), int(other["port"])
            if _pid_alive(other_pid) and other_pid != os.getpid():
                # verify it actually responds, so a zombie that holds the
                # file but crashed its HTTP thread doesn't lock us out
                import urllib.request
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{other_port}/api/runs", timeout=2)
                    return other_pid
                except Exception:
                    pass  # responds to nothing -> treat as dead
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        # stale lock
        try:
            os.remove(lock_path)
        except OSError:
            pass

    with open(lock_path, "w") as f:
        json.dump({"pid": os.getpid(), "port": port, "started": time.time()}, f)
    return None


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform liveness check."""
    if pid <= 0:
        return False
    if platform.system() == "Windows":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True)
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)   # signal 0 = existence probe on POSIX
        return True
    except OSError:
        return False


def release_lock():
    lock_path = os.path.join(RUNS_DIR, ".monitor.lock")
    try:
        if os.path.exists(lock_path):
            with open(lock_path) as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(lock_path)
    except (OSError, json.JSONDecodeError):
        pass


def main():
    ap = argparse.ArgumentParser(description="K3-Edu live training dashboard")
    ap.add_argument("--run", help="path to run dir (default: newest in runs/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    # ---- single-instance guard ----
    existing = acquire_lock(args.port)
    if existing is not None:
        print(f"A monitor server is already running (PID {existing}). "
              f"Only one instance is allowed.\n"
              f"Open http://127.0.0.1:{args.port}/ or stop the other "
              f"instance first.")
        sys.exit(1)
    import atexit
    atexit.register(release_lock)

    Handler.run_dir = find_run_dir(args.run)
    if Handler.run_dir:
        print(f"Serving run: {Handler.run_dir}")
    else:
        print("No run found yet - dashboard will wait for one. "
              "Start training to create runs/run_*/metrics.jsonl")

    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url}")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        release_lock()
        server.server_close()


if __name__ == "__main__":
    main()
