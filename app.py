"""Local web app for the stock signal pipeline.

Run it with:

    python app.py

then open http://127.0.0.1:5000 in your browser (it opens automatically).

The UI lets you type a ticker and press Train / Get Signal / Retrain. Long jobs
run in a background thread so the page stays responsive and streams the log.
"""

from __future__ import annotations

import io
import json
import re
import threading
import traceback
import webbrowser
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request

from stockpipe.config import Config
from stockpipe.pipeline import predict as run_predict
from stockpipe.pipeline import retrain as run_retrain
from stockpipe.pipeline import train as run_train

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
ARTIFACTS_ROOT = BASE_DIR / "artifacts"

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _safe_name(ticker: str) -> str:
    """Filesystem-safe folder name for a ticker (e.g. 'BRK-B', 'BTC-USD')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker).upper()


def config_for(ticker: str, start: Optional[str] = None) -> Config:
    """Load config.yaml and point it at a per-ticker artifacts folder."""
    cfg = Config.load(CONFIG_PATH)
    ticker = (ticker or cfg.data.ticker).strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty.")
    cfg.data.ticker = ticker
    if start:
        cfg.data.start = start
    cfg.paths.artifacts_dir = str(ARTIFACTS_ROOT / _safe_name(ticker))
    return cfg


# --------------------------------------------------------------------------- #
# Background job plumbing
# --------------------------------------------------------------------------- #
class Job:
    """State of a single background pipeline run."""

    def __init__(self, action: str, ticker: str):
        self.action = action
        self.ticker = ticker
        self.log: list[str] = []
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.running = True
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.log.append(line)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "action": self.action,
                "ticker": self.ticker,
                "log": list(self.log),
                "result": self.result,
                "error": self.error,
                "running": self.running,
            }


class _JobLogStream(io.TextIOBase):
    """Captures ``print`` output from the pipeline into the job log."""

    def __init__(self, job: Job):
        self._job = job
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._job.append(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._job.append(self._buffer)
            self._buffer = ""


_current_job: Optional[Job] = None
_job_lock = threading.Lock()

ACTIONS = {"train": run_train, "retrain": run_retrain, "predict": run_predict}


def _run_job(job: Job, cfg: Config) -> None:
    stream = _JobLogStream(job)
    try:
        with redirect_stdout(stream):
            if job.action == "predict":
                result = run_predict(cfg, verbose=False)
            else:
                result = ACTIONS[job.action](cfg)
        stream.flush()
        job.result = result
        job.append(f"[done] {job.action} finished for {job.ticker}.")
    except Exception as exc:  # surfaced in the UI
        stream.flush()
        job.error = str(exc) or exc.__class__.__name__
        job.append(f"[error] {job.error}")
        traceback.print_exc()
    finally:
        job.running = False


def _start_job(action: str, ticker: str, start: Optional[str]) -> tuple[bool, str]:
    global _current_job
    with _job_lock:
        if _current_job is not None and _current_job.running:
            return False, f"A {_current_job.action} job is already running."
        try:
            cfg = config_for(ticker, start)
        except ValueError as exc:
            return False, str(exc)
        job = Job(action, cfg.data.ticker)
        job.append(f"[start] {action} for {cfg.data.ticker}")
        _current_job = job
    threading.Thread(target=_run_job, args=(job, cfg), daemon=True).start()
    return True, "started"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    cfg = Config.load(CONFIG_PATH)
    return render_template(
        "index.html",
        default_ticker=cfg.data.ticker,
        buy_threshold=cfg.signal.buy_threshold,
        sell_threshold=cfg.signal.sell_threshold,
    )


@app.post("/api/<action>")
def start(action: str):
    if action not in ACTIONS:
        return jsonify({"ok": False, "message": f"Unknown action {action!r}"}), 404
    payload = request.get_json(silent=True) or {}
    ok, message = _start_job(action, payload.get("ticker", ""), payload.get("start"))
    return jsonify({"ok": ok, "message": message}), (200 if ok else 409)


@app.get("/api/status")
def status():
    job = _current_job
    if job is None:
        return jsonify({"idle": True})
    return jsonify({"idle": False, **job.snapshot()})


@app.get("/api/model")
def model_info():
    """Report whether a trained model exists for a ticker."""
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"trained": False})
    meta_path = ARTIFACTS_ROOT / _safe_name(ticker) / "metadata.json"
    if not meta_path.exists():
        return jsonify({"trained": False, "ticker": ticker})
    metadata = json.loads(meta_path.read_text())
    return jsonify(
        {
            "trained": True,
            "ticker": ticker,
            "best_model": metadata.get("best_model"),
            "trained_at": metadata.get("trained_at"),
            "selection_metric": metadata.get("selection_metric"),
            "metrics": metadata.get("metrics"),
        }
    )


def main() -> None:
    host, port = "127.0.0.1", 5000
    url = f"http://{host}:{port}"
    print(f"\n  Stock Signal Pipeline UI -> {url}\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
