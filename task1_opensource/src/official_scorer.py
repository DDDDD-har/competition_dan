#!/usr/bin/env python3
"""Thin official ScorerClient wrapper for Task 1 inspection.

SDK package: ``orca_scorer_client`` (import ``ScorerClient``).

Fixed lifecycle skeleton (swap only ``task_id`` per task)::

    from orca_scorer_client import ScorerClient

    # 1) init: auto-start local scorer_service (<=30s), auto-clean on exit
    scorer = ScorerClient(
        team_id="...",
        team_token="...",
        robot_id="g1_omnipicker",
    )
    # 2) start attempt when robot/control loop is ready
    info = scorer.start_attempt("task1_inspection")
    try:
        ...  # your control / validation mission
    finally:
        # 3) finish attempt (score + optional central/video upload)
        summary = scorer.finish()

``OfficialTaskScorer`` mirrors this: ``connect()`` → ``start()`` → ``finish()``.
Internal 0.60 m acceptance runs in parallel with official scoring.
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCORING_CLIENT_MODULE = "orca_scorer_client"
TASK1_ID = "task1_inspection"
DEFAULT_ROBOT_ID = "g1_omnipicker"
DEFAULT_ORCA_ADDR = "localhost:50051"
DEFAULT_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_HTTP_TIMEOUT_S = 120.0
DEFAULT_CENTRAL_UPLOAD_WAIT_S = 90.0
DEFAULT_SCORING_CONFIG_PATH = Path.home() / ".config/orca_scoring/orca_scoring.yaml"


def load_scoring_credentials(
    config_path: Path | None = None,
) -> tuple[str | None, str | None]:
    """Read team credentials from ``orca_scoring.yaml`` when env/CLI omit them."""
    path = config_path or Path(
        os.environ.get("ORCA_SCORING_CONFIG", str(DEFAULT_SCORING_CONFIG_PATH))
    )
    if not path.is_file():
        return None, None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None
    team_id = data.get("team_id")
    team_token = data.get("team_token")
    return (
        str(team_id) if team_id else None,
        str(team_token) if team_token else None,
    )


def apply_scoring_env(
    *,
    orca_addr: str = DEFAULT_ORCA_ADDR,
    base_url: str = DEFAULT_BASE_URL,
    team_id: str | None = None,
    team_token: str | None = None,
    record_video: bool = False,
    local_only: bool = True,
) -> None:
    """Align local client env; do not override official attempt timeout."""
    os.environ["ORCA_SCORING_ORCA_ADDR"] = orca_addr
    os.environ["ORCA_SCORING_BASE_URL"] = base_url
    os.environ.pop("ORCA_SCORING_MAX_ATTEMPT_DURATION", None)
    if team_id:
        os.environ["ORCA_SCORING_TEAM_ID"] = team_id
    if team_token:
        os.environ["ORCA_SCORING_TEAM_TOKEN"] = team_token
    if local_only:
        os.environ["ORCA_SCORING_SERVER_URL"] = ""
    else:
        os.environ.pop("ORCA_SCORING_SERVER_URL", None)
    if record_video:
        os.environ["ORCA_SCORING_VIDEO_ENABLED"] = "1"
        os.environ["ORCA_SCORING_USE_SCREEN_CAPTURE"] = "true"
    else:
        os.environ["ORCA_SCORING_VIDEO_ENABLED"] = "0"
        os.environ["ORCA_SCORING_USE_SCREEN_CAPTURE"] = "false"
        os.environ["ORCA_SCORING_KEEP_WINDOW_ON_TOP"] = "false"


def scoring_client_import_error() -> str | None:
    try:
        from orca_scorer_client import ScorerClient  # noqa: F401
    except Exception as exc:
        return str(exc)
    return None


def verify_scoring_client() -> dict[str, Any]:
    """Quick install check for Chapter 7 scorer client setup."""
    report: dict[str, Any] = {
        "import_ok": False,
        "module": None,
        "base_url": os.environ.get("ORCA_SCORING_BASE_URL", DEFAULT_BASE_URL),
        "team_id": os.environ.get("ORCA_SCORING_TEAM_ID"),
        "local_service_alive": local_service_alive(),
        "install_hint": (
            "pip install "
            "git+https://codehub.devcloud.cn-east-3.huaweicloud.com/"
            "bbb453d36204472ca668f7c8ca786e59/competition_scoring_client.git@SouthGrid"
        ),
    }
    try:
        from orca_scorer_client import ScorerClient
    except Exception as exc:
        report["error"] = str(exc)
        return report
    report["import_ok"] = True
    report["module"] = ScorerClient.__module__
    return report


def local_service_alive(host: str = "127.0.0.1", port: int = 9000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def wait_for_scorer_upload(
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout_s: float = DEFAULT_CENTRAL_UPLOAD_WAIT_S,
) -> dict[str, Any]:
    """Ask local scorer_service to drain async central/video uploads."""
    endpoint = f"{base_url.rstrip('/')}/api/shutdown"
    request = urllib.request.Request(
        endpoint,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            if not body.strip():
                return {"upload_complete": True, "endpoint": endpoint}
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                parsed.setdefault("endpoint", endpoint)
                return parsed
            return {"upload_complete": True, "endpoint": endpoint, "raw": parsed}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "upload_complete": False,
            "endpoint": endpoint,
            "error": f"HTTP {exc.code}: {detail}",
        }
    except Exception as exc:
        return {
            "upload_complete": False,
            "endpoint": endpoint,
            "error": str(exc),
        }


class OfficialTaskScorer:
    """Best-effort official scoring. Failures never raise to the caller.

    Mirrors SDK order: ``connect()`` (ScorerClient init) → ``start()``
    (``start_attempt``) → ``finish()`` (``finish``).
    """

    def __init__(
        self,
        session: Path,
        *,
        task_id: str = TASK1_ID,
        robot_id: str = DEFAULT_ROBOT_ID,
        orca_addr: str = DEFAULT_ORCA_ADDR,
        base_url: str = DEFAULT_BASE_URL,
        team_id: str | None = None,
        team_token: str | None = None,
        record_video: bool = False,
        timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
        local_only: bool = True,
        central_upload_wait_s: float = DEFAULT_CENTRAL_UPLOAD_WAIT_S,
    ) -> None:
        self.session = session
        self.task_id = task_id
        self.robot_id = robot_id
        self.orca_addr = orca_addr
        self.base_url = base_url
        yaml_team_id, yaml_team_token = load_scoring_credentials()
        self.team_id = team_id or os.environ.get("ORCA_SCORING_TEAM_ID") or yaml_team_id
        self.team_token = (
            team_token or os.environ.get("ORCA_SCORING_TEAM_TOKEN") or yaml_team_token
        )
        self.record_video = record_video
        self.timeout_s = timeout_s
        self.local_only = local_only
        self.central_upload_wait_s = central_upload_wait_s
        self.client: Any = None
        self.connected = False
        self.started = False
        self.finished = False
        self.report: dict[str, Any] = {
            "format": "official_scorer_task1_v1",
            "sdk_module": SCORING_CLIENT_MODULE,
            "task_id": task_id,
            "robot_id": robot_id,
            "orca_addr": orca_addr,
            "base_url": base_url,
            "team_id": self.team_id,
            "record_video": record_video,
            "local_only": local_only,
            "central_upload_wait_s": central_upload_wait_s,
            "central_upload": None,
            "created_at": datetime.now().astimezone().isoformat(),
            "warnings": [],
            "connect": None,
            "start": None,
            "finish": None,
            "error": None,
        }

    def _warn(self, message: str) -> None:
        self.report["warnings"].append(message)
        print(f"[official-scorer] warning: {message}", flush=True)

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "team_id": self.team_id,
            "team_token": self.team_token,
            "robot_id": self.robot_id,
            "orca_addr": self.orca_addr,
            "base_url": self.base_url,
            "keep_window_on_top": self.record_video,
            "timeout": self.timeout_s,
        }
        if self.local_only:
            kwargs["server_url"] = ""
        return kwargs

    def connect(self) -> bool:
        """Create ``ScorerClient``; SDK auto-starts scorer_service (<=30s)."""
        if self.connected and self.client is not None:
            return True
        apply_scoring_env(
            orca_addr=self.orca_addr,
            base_url=self.base_url,
            team_id=self.team_id,
            team_token=self.team_token,
            record_video=self.record_video,
            local_only=self.local_only,
        )
        if self.local_only:
            self._warn(
                "local-only scoring: central server upload disabled "
                "(ORCA_SCORING_SERVER_URL cleared)"
            )
        if local_service_alive():
            self._warn(
                "port 9000 already has a listener; ScorerClient will reuse it. "
                "Clear leftover scorer_service if it was started with unofficial flags."
            )
        try:
            from orca_scorer_client import ScorerClient
        except Exception as exc:
            self.report["error"] = f"import ScorerClient failed: {exc}"
            self._warn(self.report["error"])
            self.write()
            return False
        try:
            self.client = ScorerClient(**self._client_kwargs())
            self.connected = True
            self.report["connect"] = {
                "sdk_module": SCORING_CLIENT_MODULE,
                "team_id": self.team_id,
                "robot_id": self.robot_id,
                "base_url": self.base_url,
                "local_only": self.local_only,
            }
            self.report["connected_at"] = datetime.now().astimezone().isoformat()
            print(
                f"[official-scorer] connected via {SCORING_CLIENT_MODULE} "
                f"(task_id={self.task_id}, robot={self.robot_id})",
                flush=True,
            )
            self.write()
            return True
        except Exception as exc:
            self.report["error"] = f"ScorerClient init failed: {exc}"
            self.report["traceback"] = traceback.format_exc()
            self._warn(self.report["error"])
            self.write()
            return False

    def _sync_attempt_id(self, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("attempt_id"):
            self.report["attempt_id"] = payload["attempt_id"]

    def start(self) -> bool:
        """``start_attempt(task_id)`` after robot is ready to move."""
        if not self.connect():
            return False
        if self.started:
            return True
        try:
            started = self.client.start_attempt(self.task_id)
            self.report["start"] = started
            self._sync_attempt_id(started)
            if isinstance(started, dict) and started.get("error"):
                self.report["error"] = started["error"]
                self._warn(f"start_attempt failed: {started['error']}")
                self.write()
                return False
            self.started = True
            self.report["started_at"] = datetime.now().astimezone().isoformat()
            print(
                f"[official-scorer] started {self.task_id} "
                f"attempt={started.get('attempt_id') if isinstance(started, dict) else started}",
                flush=True,
            )
            self.write()
            return True
        except Exception as exc:
            self.report["error"] = f"start_attempt failed: {exc}"
            self.report["traceback"] = traceback.format_exc()
            self._warn(self.report["error"])
            self.write()
            return False

    def finish(self) -> dict[str, Any]:
        """``finish()`` then wait for central/video upload when enabled."""
        if self.finished:
            return self.report
        if self.client is None or not self.started:
            if self.report.get("error") is None and not self.started:
                self.report.setdefault(
                    "skipped_reason",
                    self.report.get("error") or "attempt was not started",
                )
            self.write()
            self.finished = True
            return self.report
        try:
            self.report["finish_called_at"] = datetime.now().astimezone().isoformat()
            summary = self.client.finish()
            self.report["finish"] = summary
            self._sync_attempt_id(summary)
            if isinstance(summary, dict) and summary.get("team_id"):
                self.report["team_id"] = summary["team_id"]
            if isinstance(summary, dict) and summary.get("error"):
                self._warn(f"finish failed: {summary['error']}")
                if self.report.get("error") is None:
                    self.report["error"] = summary["error"]
            else:
                score = summary.get("score") if isinstance(summary, dict) else None
                max_score = summary.get("max_score") if isinstance(summary, dict) else None
                print(
                    f"[official-scorer] finished {self.task_id}: {score}/{max_score}",
                    flush=True,
                )
            if not self.local_only:
                print(
                    f"[official-scorer] waiting up to {self.central_upload_wait_s:.0f}s "
                    "for central score/video upload...",
                    flush=True,
                )
                upload_status = wait_for_scorer_upload(
                    self.base_url,
                    timeout_s=self.central_upload_wait_s,
                )
                self.report["central_upload"] = upload_status
                if upload_status.get("upload_complete"):
                    print("[official-scorer] central upload drain complete", flush=True)
                else:
                    self._warn(
                        "central upload may be incomplete: "
                        + str(upload_status.get("error") or upload_status)
                    )
                time.sleep(1.0)
        except Exception as exc:
            self.report["error"] = f"finish failed: {exc}"
            self.report["traceback"] = traceback.format_exc()
            self._warn(self.report["error"])
        self.report["finished_at"] = datetime.now().astimezone().isoformat()
        self.finished = True
        self.write()
        return self.report

    def write(self) -> Path:
        path = self.session / "official_score.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        return path
