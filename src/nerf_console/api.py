"""FastAPI boundary for the local NeRF research console."""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import psutil
import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nerf_console_diagnostics import (
    camera_geometry, hierarchical_sampling_artifacts, list_reconstructions,
    positional_encoding_info, read_evaluations, read_run_metadata,
)

from .manager import ExperimentManager, PROJECT_ROOT, read_csv
from .ray_diagnostics import inspect_ray


class CreateRun(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True
    label: str | None = None


class EvaluateRun(BaseModel):
    checkpoint: int
    split: str = "val"
    views: list[int] = Field(default_factory=lambda: [0])
    device: str = "auto"
    render_chunk_size: int | None = None


class StartRun(BaseModel):
    device: str = "auto"


def create_app(project_root: Path = PROJECT_ROOT) -> FastAPI:
    app = FastAPI(title="NeRF Research Console", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    manager = ExperimentManager(project_root)
    app.state.manager = manager
    ray_diagnostic_slot = threading.BoundedSemaphore(1)

    def run_path(run_id: str) -> Path:
        try:
            return manager.run_path(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/health")
    def health():
        return {"status": "ok", "backend": "connected", "project_root": str(manager.project_root)}

    @app.get("/api/config/defaults")
    def defaults():
        return manager.defaults()

    @app.get("/api/scenes")
    def scenes():
        root = manager.project_root / "data" / "nerf_synthetic" / "lego"
        return {"items": [{"id": "lego", "name": "Lego", "available":
                            (root / "transforms_train.json").is_file()}]}

    @app.get("/api/system")
    def system():
        runs = manager.list_runs()
        active = next((run["id"] for run in runs if run["status"] in ("running", "stopping")), None)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        total_vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        telemetry = {"utilization_percent": None, "memory_used_bytes": None,
                     "temperature_c": None, "power_w": None}
        if gpu_name:
            try:
                probe = subprocess.run([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=False)
                if probe.returncode == 0:
                    values = [value.strip() for value in probe.stdout.splitlines()[0].split(",")]
                    if len(values) == 4:
                        def measured(raw: str) -> float | None:
                            try:
                                return float(raw)
                            except ValueError:
                                return None
                        telemetry = {"utilization_percent": measured(values[0]),
                                     "memory_used_bytes": (measured(values[1]) * 1024 * 1024
                                                           if measured(values[1]) is not None else None),
                                     "temperature_c": measured(values[2]), "power_w": measured(values[3])}
            except (OSError, subprocess.TimeoutExpired, IndexError):
                pass
        memory = psutil.virtual_memory()
        return {"backend": {"status": "ok", "python": sys.version.split()[0],
                            "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "active_run": active},
                "gpu": {"name": gpu_name, "memory_total_bytes": total_vram, **telemetry},
                "cpu": {"name": platform.processor() or None,
                        "utilization_percent": psutil.cpu_percent(interval=0.1)},
                "ram": {"used_bytes": memory.used, "total_bytes": memory.total}}

    @app.get("/api/runs")
    def list_runs():
        return {"items": manager.list_runs()}

    @app.post("/api/runs", status_code=201)
    def create_run(payload: CreateRun):
        try:
            return manager.create_run(payload.config, auto_start=payload.auto_start, label=payload.label)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run_path(run_id)
        return manager.get_run(run_id)

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str, start: int = Query(0, ge=0), limit: int = Query(60000, ge=1, le=100000)):
        path = run_path(run_id)
        rows = read_csv(path / "metrics.csv")
        return {"rows": rows[start:start + limit], "total": len(rows),
                "latest": rows[-1] if rows else None,
                "semantics": {"psnr": "fine-network PSNR from a sampled training ray batch; not full-view validation PSNR",
                              "ssim": "not recorded during training",
                              "lpips": "not recorded during training"}}

    @app.get("/api/runs/{run_id}/checkpoints")
    def checkpoints(run_id: str):
        run_path(run_id)
        return manager.checkpoints(run_id)

    @app.get("/api/runs/{run_id}/reconstructions")
    def reconstructions(run_id: str):
        path = run_path(run_id)
        groups = list_reconstructions(path)
        items = []
        for group in groups:
            for kind, relative in group.get("images", {}).items():
                if relative:
                    items.append({"iteration": group["iteration"], "split": group["split"],
                                  "view_index": group["view_index"], "kind": kind,
                                  "url": manager.artifact_url(run_id, Path(relative)),
                                  "metrics": group.get("metrics")})
        # Diagnostic training previews have no full-view PSNR/SSIM/LPIPS.
        for file in sorted((path / "renders").glob("*_iter_*.png")):
            stem = file.stem
            try:
                kind, iteration = stem.rsplit("_iter_", 1)
                items.append({"iteration": int(iteration), "split": "val",
                              "view_index": 0, "kind": kind,
                              "url": manager.artifact_url(run_id, file.relative_to(path)),
                              "metrics": None, "diagnostic_preview": True})
            except ValueError:
                continue
        return {"items": items, "available": bool(items)}

    @app.get("/api/runs/{run_id}/renders")
    def renders(run_id: str):
        return reconstructions(run_id)

    @app.get("/api/runs/{run_id}/evaluations")
    def evaluations(run_id: str):
        result = read_evaluations(run_path(run_id))
        return {**result, "items": result.get("per_view", [])}

    @app.get("/api/runs/{run_id}/sampling")
    def sampling(run_id: str):
        result = hierarchical_sampling_artifacts(run_path(run_id), manager.project_root)
        result["on_demand_available"] = bool(manager.checkpoints(run_id)["items"])
        result["ray_endpoint"] = f"/api/runs/{run_id}/sampling/ray"
        for figure in result.get("static_figures", []):
            figure["url"] = f"/api/validation-artifacts/{figure['path']}"
        return result

    @app.get("/api/runs/{run_id}/sampling/ray")
    def sampling_ray(run_id: str, checkpoint: int = Query(..., ge=0),
                     split: str = Query("val"), view_index: int = Query(0, ge=0),
                     x: int = Query(..., ge=0), y: int = Query(..., ge=0)):
        path = run_path(run_id)
        if not ray_diagnostic_slot.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="another ray diagnostic is running")
        try:
            return inspect_ray(manager.project_root, path, checkpoint=checkpoint,
                               split=split, view_index=view_index, x=x, y=y)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, OSError, yaml.YAMLError) as exc:
            raise HTTPException(status_code=409, detail=f"ray inspection failed: {exc}") from exc
        finally:
            ray_diagnostic_slot.release()

    @app.get("/api/validation-artifacts/{relative:path}")
    def validation_artifact(relative: str):
        base = (manager.project_root / "artifacts").resolve()
        target = (manager.project_root / relative).resolve()
        if not target.is_relative_to(base) or not target.is_file() or target.suffix.lower() != ".png":
            raise HTTPException(status_code=404, detail="validation figure unavailable")
        return FileResponse(target)

    @app.get("/api/runs/{run_id}/metadata")
    def metadata(run_id: str):
        return read_run_metadata(run_path(run_id))

    @app.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str):
        path = run_path(run_id)
        files = []
        for folder in ("artifacts", "evaluation", "evaluations"):
            base = path / folder
            if base.is_dir():
                for item in base.rglob("*"):
                    if item.is_file() and item.suffix.lower() in (".png", ".jpg", ".json", ".csv", ".txt"):
                        files.append({"path": item.relative_to(path).as_posix(),
                                      "size_bytes": item.stat().st_size,
                                      "url": manager.artifact_url(run_id, item.relative_to(path))})
        return {"items": files}

    @app.get("/api/runs/{run_id}/logs")
    def logs(run_id: str, tail: int = Query(300, ge=1, le=10000)):
        path = run_path(run_id)
        result = {}
        for log in [path / "train.log", *sorted((path / "logs").glob("*.log"))]:
            if log.is_file():
                with log.open("r", encoding="utf-8", errors="replace") as stream:
                    from collections import deque
                    result[log.name] = list(deque(stream, maxlen=tail))
        return {"files": result, "lines": [line.rstrip("\n") for lines in result.values() for line in lines]}

    @app.get("/api/runs/{run_id}/files/{relative:path}")
    def file(run_id: str, relative: str):
        path = run_path(run_id)
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or not target.is_file():
            raise HTTPException(status_code=404, detail="artifact unavailable")
        if target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".json", ".csv", ".txt", ".log", ".yaml", ".pt"):
            raise HTTPException(status_code=403, detail="artifact type is not served")
        return FileResponse(target)

    @app.post("/api/runs/{run_id}/start")
    def start(run_id: str, payload: StartRun = StartRun()):
        run_path(run_id)
        try:
            return manager.start(run_id, device=payload.device)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str):
        run_path(run_id)
        try:
            return manager.stop(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, payload: StartRun = StartRun()):
        run_path(run_id)
        try:
            return manager.start(run_id, resume=True, device=payload.device)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate(run_id: str, payload: EvaluateRun | None = None):
        run_path(run_id)
        if payload is None:
            checkpoints = manager.checkpoints(run_id)["items"]
            if not checkpoints:
                raise HTTPException(status_code=422, detail="no checkpoint available for evaluation")
            payload = EvaluateRun(checkpoint=checkpoints[-1]["iteration"])
        try:
            return manager.evaluate(run_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/replay")
    def replay(run_id: str):
        path = run_path(run_id)
        with manager._lock:
            if run_id in manager._replays:
                raise HTTPException(status_code=409, detail="replay is already running")
            manager._replays.add(run_id)
        def feed():
            try:
                rows = read_csv(path / "metrics.csv")
                manager._publish(run_id, "replay_started", label="REPLAY MODE — NOT LIVE SCIENTIFIC TRAINING")
                step = max(1, len(rows) // 300)
                for row in rows[::step]:
                    manager._publish(run_id, "metric_update", metrics=row, replay=True)
                    time.sleep(0.04)
                manager._publish(run_id, "replay_completed", replay=True)
            finally:
                with manager._lock:
                    manager._replays.discard(run_id)
        threading.Thread(target=feed, daemon=True).start()
        return {"status": "replay", "label": "REPLAY MODE — NOT LIVE SCIENTIFIC TRAINING"}

    @app.get("/api/scenes/{scene}/cameras")
    def cameras(scene: str, split: str | None = None, start: int = 0, limit: int = 100,
                pixel_x: int | None = None, pixel_y: int | None = None):
        if scene != "lego" or (split is not None and split not in ("train", "val", "test")):
            raise HTTPException(status_code=404, detail="scene or split unavailable")
        if (pixel_x is None) != (pixel_y is None):
            raise HTTPException(status_code=422, detail="pixel_x and pixel_y must be supplied together")
        return camera_geometry(manager.project_root / "data" / "nerf_synthetic", scene,
                               split=split, start=start, limit=limit,
                               pixel_xy=(pixel_x, pixel_y) if pixel_x is not None else None)

    @app.get("/api/positional-encoding/{L}")
    def positional_encoding(L: int):
        if not 0 <= L <= 15:
            raise HTTPException(status_code=422, detail="L must be 0–15")
        return positional_encoding_info(L)

    @app.websocket("/ws/runs/{run_id}")
    async def websocket_run(websocket: WebSocket, run_id: str):
        try:
            manager.run_path(run_id)
        except (ValueError, FileNotFoundError):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        seq = 0
        try:
            while True:
                for event in manager.events_after(run_id, seq):
                    await websocket.send_json(event)
                    seq = event["seq"]
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            pass

    # The desktop shell loads the built frontend from this same local origin.
    # Scientific APIs and the WebSocket keep their existing paths and behavior.
    frontend_dist = Path(os.environ["NERF_FRONTEND_DIST"]).resolve() if os.environ.get("NERF_FRONTEND_DIST") else manager.project_root / "console_frontend" / "dist"

    @app.get("/", include_in_schema=False)
    def desktop_frontend():
        index = frontend_dist / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="desktop frontend has not been built")
        return FileResponse(index)

    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets", check_dir=False), name="desktop-assets")

    return app


app = create_app()
