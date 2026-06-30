#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Turbo sliding-window Video2World driver for NVIDIA Cosmos-Predict2.5.

Built to stay close to the official `examples/inference.py` path while adding a
sliding future-prediction schedule:

    1 -> target, 1-2 -> next target, ..., 1-5 -> next target,
    2-6 -> next target, 3-7 -> next target, ...

The first 2-4 frame windows are left-padded to the five pixel frames expected by
Cosmos-Predict2.5 Video2World. After the warmup, every generation conditions on
the latest five source frames.

Main speedups compared with the earlier wrapper:

1. Short temporal budget:
   Overrides `model.config.state_t` so Cosmos only runs enough latent time steps
   to contain the requested target frame instead of the native 77-frame clip.

2. Direct frame output:
   For target-frame assembly, the worker extracts target PNGs directly from the
   returned video tensor. It avoids saving hundreds of short mp4 clips and avoids
   ffmpeg reading those clips back only to take one frame.

3. Prompt embedding cache:
   The official pipeline recomputes the same prompt and negative-prompt text
   embeddings for every sample. This wrapper monkey-patches the text encoder with
   an in-process cache keyed by caption text.

4. Tensor conditioning mode:
   Instead of writing one 5-frame mp4 per sliding window, the parent extracts the
   source video frames once. The worker builds the padded conditioning tensor
   directly and passes it to the official Video2World pipeline.

5. Data-parallel sharding:
   For models that fit on one GPU, `--parallelism data` launches one independent
   worker per GPU and splits windows across GPUs. For short target-frame jobs this
   often beats context parallelism because there is less inter-GPU communication.

6. Low-VRAM hygiene:
   Defaults to model/tokenizer/text-encoder offload, bounded/disable-able prompt
   cache, aggressive per-window garbage collection, CUDA cache trimming, and
   optional worker process recycling to defeat long-run allocator fragmentation.

Typical H200 x8 high-quality use from the Cosmos repo root:

    python /path/to/sliding_video2world_future.py \
      --input-video assets/base/sand_mining.mp4 \
      --output-video outputs/sliding_pred_14b.mp4 \
      --prompt "The scene continues with physically realistic motion." \
      --model 14B/post-trained --num-gpus 8 --turbo-profile exact

Approximate high-speed 2B mode:

    python /path/to/sliding_video2world_future.py \
      --input-video assets/base/sand_mining.mp4 \
      --output-video outputs/sliding_pred_veryfast.mp4 \
      --prompt "The scene continues with physically realistic motion." \
      --window-schedule fixed --cond-frames 5 \
      --model 2B/post-trained --num-gpus 2 --turbo-profile veryfast

Important:
  - Do not launch this wrapper itself with torchrun. Use --num-gpus N.
  - `--turbo-profile exact` keeps the exact warmup/sliding-window semantics.
  - fast/veryfast/insane reduce the number of Cosmos generations by extracting
    multiple target frames from one generated clip; this is an approximation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Literal


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def shlex_join(cmd: Iterable[object]) -> str:
    return shlex.join(str(x) for x in cmd)


def run_cmd(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    str_cmd = [str(x) for x in cmd]
    eprint(f"$ {shlex_join(str_cmd)}")
    return subprocess.run(
        str_cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def require_executable(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")
    return exe


def parse_fps(value: str | int | float | None) -> float:
    if value is None:
        return 16.0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if not value or value == "0/0":
        return 16.0
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("b must be positive")
    return -(-a // b)


def safe_name(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(out).strip("_") or "sample"


def rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def latent_to_pixel_frames(latent_t: int, *, temporal_compression_factor: int = 4) -> int:
    if latent_t < 1:
        raise ValueError("latent_t must be >= 1")
    return temporal_compression_factor * (latent_t - 1) + 1


def pixel_to_latent_frames_ceil(pixel_frames: int, *, temporal_compression_factor: int = 4) -> int:
    if pixel_frames < 1:
        raise ValueError("pixel_frames must be >= 1")
    return ceil_div(pixel_frames - 1, temporal_compression_factor) + 1


def choose_runtime_state_t(
    *,
    min_required_pixel_frames: int,
    budget_parallel_size: int,
    temporal_compression_factor: int,
    align_latent_to_parallel: bool,
    manual_state_t: int | None,
) -> int:
    if manual_state_t is not None:
        if manual_state_t < 1:
            raise ValueError("manual_state_t must be >= 1")
        return manual_state_t

    latent_t = pixel_to_latent_frames_ceil(
        min_required_pixel_frames,
        temporal_compression_factor=temporal_compression_factor,
    )
    if align_latent_to_parallel and budget_parallel_size > 1:
        latent_t = ceil_div(latent_t, budget_parallel_size) * budget_parallel_size
    return latent_t


def parse_gpu_ids(text: str | None, num_gpus: int) -> list[str]:
    if text:
        ids = [x.strip() for x in text.split(",") if x.strip()]
        if not ids:
            raise SystemExit("--gpu-ids was provided but no GPU ids were parsed")
        return ids
    return [str(i) for i in range(num_gpus)]


def chunk_contiguous(items: list[Any], n: int) -> list[list[Any]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not items:
        return [[] for _ in range(n)]
    chunk_size = ceil_div(len(items), n)
    return [items[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float | None


@dataclass(frozen=True)
class WindowSpec:
    index: int
    start_frame: int
    end_frame: int
    output_start_index: int
    cond_frames: int
    effective_cond_frames: int
    target_index: int
    input_path: str
    json_path: str
    sample_name: str
    inference_type: Literal["image2world", "video2world"]
    seed: int


@dataclass(frozen=True)
class WindowPlan:
    output_index: int
    start_frame: int
    cond_frames: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.cond_frames - 1


@dataclass(frozen=True)
class Manifest:
    repo_root: str
    created_at_unix: float
    input_video: str
    output_video: str
    work_dir: str
    cosmos_output_dir: str
    params_dir: str
    clips_dir: str
    frames_dir: str
    source_frames_dir: str
    prompt: str
    model: str
    num_gpus: int
    parallelism: Literal["context", "data"]
    conditioning_mode: Literal["tensor", "clips"]
    direct_frame_output: bool
    disable_guardrails_effective: bool
    fps: float
    output_fps: float
    window_schedule: Literal["fixed", "warmup5"]
    cond_frames: int
    effective_cond_frames: int
    stride: int
    future_offset: int
    num_output_frames: int
    native_num_output_frames: int
    runtime_state_t: int
    temporal_compression_factor: int
    target_index: int
    max_target_index: int
    samples_per_generation: int
    desired_window_count: int
    aggregation: str
    resume: bool
    cuda_empty_cache_every: int
    use_tf32: bool
    memory_profile: Literal["speed", "balanced", "lowvram", "ultralowvram"]
    prompt_cache: bool
    prompt_cache_device: Literal["cpu", "cuda"]
    prompt_cache_max_entries: int
    clear_prompt_cache_every: int
    cleanup_cpu_offload_after_each: bool
    gc_collect_every: int
    cuda_ipc_collect: bool
    recycle_worker_every: int
    cuda_alloc_conf: str
    windows: list[WindowSpec]
    setup: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["windows"] = [asdict(w) for w in self.windows]
        return data

    @classmethod
    def from_path(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text())
        raw_windows = data.pop("windows")
        data.setdefault("window_schedule", "fixed")
        default_cond_frames = int(data["cond_frames"])
        default_effective_cond_frames = int(data["effective_cond_frames"])
        default_target_index = int(data["target_index"])
        windows = []
        for w in raw_windows:
            w.setdefault("cond_frames", default_cond_frames)
            w.setdefault("effective_cond_frames", default_effective_cond_frames)
            w.setdefault("target_index", default_target_index)
            windows.append(WindowSpec(**w))
        return cls(windows=windows, **data)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_jsonable(), ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------
# ffmpeg/ffprobe layer
# ---------------------------------------------------------------------------


def ffprobe_video(path: Path) -> VideoInfo:
    require_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd, capture=True)
    data = json.loads(proc.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream found in {path}")
    stream = data["streams"][0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))

    duration = None
    for raw in (stream.get("duration"), data.get("format", {}).get("duration")):
        if raw not in (None, "N/A"):
            try:
                duration = float(raw)
                break
            except ValueError:
                pass

    frame_count = None
    nb_frames = stream.get("nb_frames")
    if nb_frames not in (None, "N/A", ""):
        try:
            frame_count = int(nb_frames)
        except ValueError:
            frame_count = None

    if frame_count is None:
        count_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        count_proc = run_cmd(count_cmd, capture=True)
        text = count_proc.stdout.strip()
        if text and text != "N/A":
            frame_count = int(text)
        elif duration is not None and fps > 0:
            frame_count = int(round(duration * fps))
        else:
            raise RuntimeError(f"Could not determine frame count for {path}")

    return VideoInfo(str(path), width, height, fps, frame_count, duration)


def ffmpeg_extract_source_frames(
    input_video: Path,
    frames_dir: Path,
    *,
    frame_count: int,
    resume: bool,
    image_ext: Literal["png", "jpg"],
    jpg_quality: int,
) -> None:
    """Extract source frames once. File src_00000001.* corresponds to frame index 0."""
    require_executable("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / f"src_%08d.{image_ext}"

    existing = list(frames_dir.glob(f"src_*.{image_ext}"))
    if resume and len(existing) >= frame_count:
        eprint(f"Reusing source frame cache: {frames_dir} ({len(existing)} files)")
        return

    for p in existing:
        p.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
    ]
    if image_ext == "jpg":
        cmd += ["-q:v", str(jpg_quality)]
    cmd += [str(pattern)]
    run_cmd(cmd)

    produced = len(list(frames_dir.glob(f"src_*.{image_ext}")))
    if produced < frame_count:
        raise RuntimeError(f"Source frame cache produced {produced} frames; expected at least {frame_count}")
    eprint(f"Extracted source frame cache: {frames_dir} ({produced} frames)")


def ffmpeg_extract_image_frame(
    input_video: Path,
    frame_index: int,
    output_image: Path,
    *,
    quality: int = 2,
) -> None:
    require_executable("ffmpeg")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_index})"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
        str(output_image),
    ]
    run_cmd(cmd)
    if not output_image.exists() or output_image.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract frame {frame_index} from {input_video}")


def ffmpeg_extract_video_window(
    input_video: Path,
    start_frame: int,
    end_frame: int,
    output_video: Path,
    *,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    require_executable("ffmpeg")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    vf = f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/({fps}*TB)"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-an",
        "-r",
        f"{fps:.6f}",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError:
        if codec == "libx264":
            eprint("libx264 failed; retrying temporary clip with mpeg4.")
            ffmpeg_extract_video_window(
                input_video,
                start_frame,
                end_frame,
                output_video,
                fps=fps,
                codec="mpeg4",
                crf=crf,
                preset=preset,
            )
            return
        raise

    expected = end_frame - start_frame + 1
    info = ffprobe_video(output_video)
    if info.frame_count != expected:
        raise RuntimeError(f"{output_video} has {info.frame_count} frames; expected {expected}")


def ffmpeg_concat_images_to_video(
    frames_dir: Path,
    output_video: Path,
    *,
    fps: float,
    pattern: str = "frame_%08d.png",
    codec: str = "libx264",
    crf: int = 16,
    preset: str = "veryfast",
) -> None:
    require_executable("ffmpeg")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(frames_dir / pattern)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        input_pattern,
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError:
        if codec == "libx264":
            eprint("libx264 failed while assembling; retrying final video with mpeg4.")
            ffmpeg_concat_images_to_video(
                frames_dir,
                output_video,
                fps=fps,
                pattern=pattern,
                codec="mpeg4",
                crf=crf,
                preset=preset,
            )
            return
        raise


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def build_window_starts(
    *,
    frame_count: int,
    cond_frames: int,
    stride: int,
    start_frame: int,
    stop_frame: int | None,
    max_windows: int | None,
) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if cond_frames <= 0:
        raise ValueError("cond_frames must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    stop = frame_count if stop_frame is None else min(stop_frame, frame_count)
    last_start = stop - cond_frames
    if last_start < start_frame:
        raise ValueError(
            f"Not enough frames: frame_count={frame_count}, cond_frames={cond_frames}, "
            f"start_frame={start_frame}, stop_frame={stop}"
        )
    starts = list(range(start_frame, last_start + 1, stride))
    if max_windows is not None:
        starts = starts[:max_windows]
    return starts


def effective_conditioning_frames(cond_frames: int) -> int:
    if cond_frames == 1:
        return 1
    if 2 <= cond_frames <= 5:
        return 5
    raise ValueError("cond_frames must be in [1, 5]")


def target_index_for_conditioning(cond_frames: int, future_offset: int, override: int | None) -> int:
    if override is not None:
        return override
    return effective_conditioning_frames(cond_frames) + future_offset - 1


def build_window_plans(
    *,
    frame_count: int,
    cond_frames: int,
    stride: int,
    start_frame: int,
    stop_frame: int | None,
    max_windows: int | None,
    schedule: Literal["fixed", "warmup5"],
) -> list[WindowPlan]:
    if schedule == "fixed":
        return [
            WindowPlan(output_index=i, start_frame=start, cond_frames=cond_frames)
            for i, start in enumerate(
                build_window_starts(
                    frame_count=frame_count,
                    cond_frames=cond_frames,
                    stride=stride,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    max_windows=max_windows,
                )
            )
        ]

    if schedule != "warmup5":
        raise ValueError(f"Unknown window schedule: {schedule}")
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if cond_frames < 1 or cond_frames > 5:
        raise ValueError("--window-schedule warmup5 requires --cond-frames in [1, 5]")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    stop = frame_count if stop_frame is None else min(stop_frame, frame_count)
    if stop <= start_frame:
        raise ValueError(
            f"Not enough frames: frame_count={frame_count}, start_frame={start_frame}, stop_frame={stop}"
        )

    plans: list[WindowPlan] = []
    output_index = 0
    for end in range(start_frame, stop, stride):
        current_cond = min(cond_frames, end - start_frame + 1)
        start = end - current_cond + 1
        plans.append(WindowPlan(output_index=output_index, start_frame=start, cond_frames=current_cond))
        output_index += 1
        if max_windows is not None and len(plans) >= max_windows:
            break
    return plans


def make_inference_json(
    *,
    json_path: Path,
    sample_name: str,
    inference_type: Literal["image2world", "video2world"],
    input_path: Path,
    prompt: str,
    num_output_frames: int,
    seed: int,
    guidance: int,
    num_steps: int | None,
    resolution: str | None,
    negative_prompt: str | None,
    enable_autoregressive: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> None:
    payload: dict[str, Any] = {
        "inference_type": inference_type,
        "name": sample_name,
        "prompt": prompt,
        "input_path": str(input_path.resolve()),
        "num_output_frames": num_output_frames,
        "seed": seed,
        "guidance": guidance,
    }
    if num_steps is not None:
        payload["num_steps"] = num_steps
    if resolution:
        payload["resolution"] = resolution
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if enable_autoregressive:
        payload["enable_autoregressive"] = True
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            payload["chunk_overlap"] = chunk_overlap
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def create_padded_five_frame_clip(
    *,
    input_video: Path,
    start_frame: int,
    cond_frames: int,
    output_video: Path,
    tmp_dir: Path,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    if cond_frames not in (2, 3, 4):
        raise ValueError("padded conditioning only supports cond_frames=2..4")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_dir / "raw"
    seq_dir = tmp_dir / "seq"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    for j in range(cond_frames):
        ffmpeg_extract_image_frame(input_video, start_frame + j, raw_dir / f"raw_{j:02d}.png")

    pad = 5 - cond_frames
    for k in range(5):
        src_idx = 0 if k < pad else k - pad
        shutil.copy2(raw_dir / f"raw_{src_idx:02d}.png", seq_dir / f"frame_{k:08d}.png")

    ffmpeg_concat_images_to_video(
        seq_dir,
        output_video,
        fps=fps,
        pattern="frame_%08d.png",
        codec=codec,
        crf=crf,
        preset=preset,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)


def resolve_parallelism(args: argparse.Namespace) -> Literal["context", "data"]:
    if args.num_gpus <= 1:
        return "context"
    if args.parallelism != "auto":
        return args.parallelism
    # Low-VRAM mode should not assume one full model replica fits per GPU.
    # Context parallelism is slower for short clips, but it avoids data-parallel
    # replication and matches the official multi-GPU path more closely.
    if getattr(args, "memory_profile", "lowvram") in ("lowvram", "ultralowvram"):
        return "context"
    # Short-frame 2B workloads are usually throughput-limited rather than memory-limited.
    # Launching one independent 2B worker per GPU is often faster than context parallelism.
    if str(args.model).startswith("2B"):
        return "data"
    return "context"


def _default_if_none(args: argparse.Namespace, name: str, value: Any) -> None:
    if getattr(args, name) is None:
        setattr(args, name, value)


def apply_memory_profile(args: argparse.Namespace) -> None:
    profile = args.memory_profile

    if profile == "speed":
        _default_if_none(args, "offload_diffusion_model", False)
        _default_if_none(args, "offload_tokenizer", False)
        _default_if_none(args, "offload_text_encoder", False)
        _default_if_none(args, "offload_guardrail_models", True)
        _default_if_none(args, "prompt_cache", True)
        _default_if_none(args, "prompt_cache_device", "cuda")
        _default_if_none(args, "prompt_cache_max_entries", 16)
        _default_if_none(args, "clear_prompt_cache_every", 0)
        _default_if_none(args, "cuda_empty_cache_every", 0)
        _default_if_none(args, "gc_collect_every", 0)
        _default_if_none(args, "cleanup_cpu_offload_after_each", False)
        _default_if_none(args, "cuda_ipc_collect", False)
        _default_if_none(args, "recycle_worker_every", 0)
    elif profile == "balanced":
        _default_if_none(args, "offload_diffusion_model", False)
        _default_if_none(args, "offload_tokenizer", True)
        _default_if_none(args, "offload_text_encoder", True)
        _default_if_none(args, "offload_guardrail_models", True)
        _default_if_none(args, "prompt_cache", True)
        _default_if_none(args, "prompt_cache_device", "cpu")
        _default_if_none(args, "prompt_cache_max_entries", 4)
        _default_if_none(args, "clear_prompt_cache_every", 64)
        _default_if_none(args, "cuda_empty_cache_every", 4)
        _default_if_none(args, "gc_collect_every", 4)
        _default_if_none(args, "cleanup_cpu_offload_after_each", True)
        _default_if_none(args, "cuda_ipc_collect", True)
        _default_if_none(args, "recycle_worker_every", 0)
    elif profile == "lowvram":
        _default_if_none(args, "offload_diffusion_model", True)
        _default_if_none(args, "offload_tokenizer", True)
        _default_if_none(args, "offload_text_encoder", True)
        _default_if_none(args, "offload_guardrail_models", True)
        _default_if_none(args, "prompt_cache", False)
        _default_if_none(args, "prompt_cache_device", "cpu")
        _default_if_none(args, "prompt_cache_max_entries", 2)
        _default_if_none(args, "clear_prompt_cache_every", 1)
        _default_if_none(args, "cuda_empty_cache_every", 1)
        _default_if_none(args, "gc_collect_every", 1)
        _default_if_none(args, "cleanup_cpu_offload_after_each", True)
        _default_if_none(args, "cuda_ipc_collect", True)
        _default_if_none(args, "recycle_worker_every", 32)
    elif profile == "ultralowvram":
        _default_if_none(args, "offload_diffusion_model", True)
        _default_if_none(args, "offload_tokenizer", True)
        _default_if_none(args, "offload_text_encoder", True)
        _default_if_none(args, "offload_guardrail_models", True)
        _default_if_none(args, "prompt_cache", False)
        _default_if_none(args, "prompt_cache_device", "cpu")
        _default_if_none(args, "prompt_cache_max_entries", 1)
        _default_if_none(args, "clear_prompt_cache_every", 1)
        _default_if_none(args, "cuda_empty_cache_every", 1)
        _default_if_none(args, "gc_collect_every", 1)
        _default_if_none(args, "cleanup_cpu_offload_after_each", True)
        _default_if_none(args, "cuda_ipc_collect", True)
        _default_if_none(args, "recycle_worker_every", 8)
    else:
        raise ValueError(f"Unknown memory profile: {profile}")

    _default_if_none(args, "cuda_alloc_conf", "expandable_segments:True")


def apply_turbo_profile(args: argparse.Namespace) -> None:
    profile = args.turbo_profile
    if args.samples_per_generation is None:
        args.samples_per_generation = {
            "exact": 1,
            "fast": 2,
            "veryfast": 4,
            "insane": 8,
        }[profile]
    if args.num_steps is None:
        args.num_steps = {
            "exact": 35,
            "fast": 28,
            "veryfast": 20,
            "insane": 12,
        }[profile]
    if args.frame_budget is None:
        args.frame_budget = "auto"


def prepare_windows_and_manifest(args: argparse.Namespace) -> Path:
    input_video = Path(args.input_video).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()

    if not input_video.exists():
        raise SystemExit(f"Input video not found: {input_video}")
    if not repo_root.exists():
        raise SystemExit(f"Cosmos repo root not found: {repo_root}")
    if not (repo_root / "examples" / "inference.py").exists():
        raise SystemExit(f"Official examples/inference.py not found under repo root: {repo_root}")

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().resolve().read_text().strip()
    if not prompt:
        raise SystemExit("Provide --prompt or --prompt-file.")

    if args.window_schedule == "warmup5":
        if args.cond_frames < 1 or args.cond_frames > 5:
            raise SystemExit("--window-schedule warmup5 requires --cond-frames in [1, 5]")
    elif args.cond_frames not in (1, 5):
        if not args.allow_pad_cond_frames:
            raise SystemExit(
                "Use --cond-frames 1 or 5 for the official-clean fixed path. "
                "For fixed 2..4, add --allow-pad-cond-frames to left-pad into a 5-frame V2W clip."
            )
        if args.cond_frames < 2 or args.cond_frames > 4:
            raise SystemExit("--allow-pad-cond-frames only applies to cond_frames=2..4")

    effective_cond_frames = effective_conditioning_frames(args.cond_frames)

    if args.future_offset <= 0:
        raise SystemExit("--future-offset must be >= 1")
    if args.samples_per_generation < 1:
        raise SystemExit("--samples-per-generation must be >= 1")
    if args.window_schedule == "warmup5" and args.samples_per_generation > 1:
        raise SystemExit("--window-schedule warmup5 requires exact mode: use --turbo-profile exact or --samples-per-generation 1")
    if args.samples_per_generation > 1 and args.aggregation != "target_frame":
        raise SystemExit("--samples-per-generation > 1 is only supported with --aggregation target_frame")
    if args.conditioning_mode == "tensor" and args.aggregation != "target_frame":
        raise SystemExit("--conditioning-mode tensor currently supports --aggregation target_frame")
    if args.parallelism == "data" and not args.direct_frame_output:
        raise SystemExit("--parallelism data requires --direct-frame-output")

    parallelism = resolve_parallelism(args)
    budget_parallel_size = args.num_gpus if parallelism == "context" else 1

    info = ffprobe_video(input_video)
    window_plans = build_window_plans(
        frame_count=info.frame_count,
        cond_frames=args.cond_frames,
        stride=args.stride,
        start_frame=args.start_frame,
        stop_frame=args.stop_frame,
        max_windows=args.max_windows,
        schedule=args.window_schedule,
    )
    if not window_plans:
        raise SystemExit("No windows were produced. Check --start-frame, --stop-frame, and --max-windows.")

    target_index = target_index_for_conditioning(args.cond_frames, args.future_offset, args.target_index)
    max_target_index = max(
        target_index_for_conditioning(plan.cond_frames, args.future_offset, args.target_index)
        + (args.samples_per_generation - 1) * args.stride
        for plan in window_plans
    )

    native_num_output_frames = 77
    if args.num_output_frames is None:
        requested_min_frames = native_num_output_frames if args.frame_budget == "native" else max_target_index + 1
    else:
        requested_min_frames = args.num_output_frames
    if requested_min_frames <= max_target_index:
        raise SystemExit(
            f"Requested output length {requested_min_frames} is too small for max_target_index={max_target_index}. "
            f"Use at least {max_target_index + 1}."
        )

    runtime_state_t = choose_runtime_state_t(
        min_required_pixel_frames=requested_min_frames,
        budget_parallel_size=budget_parallel_size,
        temporal_compression_factor=args.temporal_compression_factor,
        align_latent_to_parallel=args.align_latent_to_parallel,
        manual_state_t=args.runtime_state_t,
    )
    num_output_frames = latent_to_pixel_frames(
        runtime_state_t,
        temporal_compression_factor=args.temporal_compression_factor,
    )
    if num_output_frames <= max_target_index:
        raise SystemExit(
            f"runtime_state_t={runtime_state_t} gives {num_output_frames} pixel frames, "
            f"but max_target_index={max_target_index}. Increase --runtime-state-t or --num-output-frames."
        )

    fps = args.fps if args.fps is not None else 16.0
    output_fps = args.output_fps if args.output_fps is not None else fps

    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else output_video.with_suffix("").with_name(output_video.stem + "_work")
    clips_dir = work_dir / "clips"
    params_dir = work_dir / "params"
    frames_dir = work_dir / "assembled_frames"
    cosmos_output_dir = work_dir / "cosmos_outputs"
    source_frames_dir = work_dir / "source_frames"

    if work_dir.exists() and args.overwrite_workdir:
        shutil.rmtree(work_dir)
    for d in (clips_dir, params_dir, frames_dir, cosmos_output_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.direct_frame_output and not args.resume and frames_dir.exists():
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()

    generation_offsets = list(range(0, len(window_plans), args.samples_per_generation))

    if args.conditioning_mode == "tensor":
        ffmpeg_extract_source_frames(
            input_video,
            source_frames_dir,
            frame_count=info.frame_count,
            resume=args.resume,
            image_ext=args.source_frame_ext,
            jpg_quality=args.source_jpg_quality,
        )

    eprint(
        f"Input: {input_video} | {info.width}x{info.height} | "
        f"{info.frame_count} frames @ {info.fps:.4f} fps"
    )
    eprint(
        f"Strategy: profile={args.turbo_profile}, memory={args.memory_profile}, parallelism={parallelism}, "
        f"conditioning={args.conditioning_mode}, direct_frame_output={args.direct_frame_output}, "
        f"guardrails_disabled={args.disable_guardrails}"
    )
    eprint(
        "VRAM hygiene: "
        f"offload_dit={args.offload_diffusion_model}, offload_tokenizer={args.offload_tokenizer}, "
        f"offload_text_encoder={args.offload_text_encoder}, prompt_cache={args.prompt_cache}, "
        f"cuda_empty_cache_every={args.cuda_empty_cache_every}, gc_collect_every={args.gc_collect_every}, "
        f"recycle_worker_every={args.recycle_worker_every}"
    )
    eprint(
        f"Preparing {len(generation_offsets)} Cosmos generations to cover {len(window_plans)} desired windows: "
        f"schedule={args.window_schedule}, max_cond_frames={args.cond_frames}, stride={args.stride}, "
        f"future_offset={args.future_offset}, target_index={target_index}, max_target_index={max_target_index}, "
        f"samples_per_generation={args.samples_per_generation}"
    )
    eprint(
        f"Frame budget: runtime_state_t={runtime_state_t}, num_output_frames={num_output_frames} "
        f"(native default={native_num_output_frames}; "
        f"{100.0 * (1.0 - num_output_frames / native_num_output_frames):.1f}% fewer pixel frames per generation)"
    )
    if args.samples_per_generation > 1:
        eprint(
            "Approximation enabled: one Cosmos generation supplies multiple assembled frames. "
            "Use --turbo-profile exact or --samples-per-generation 1 for exact sliding-window semantics."
        )
    if parallelism == "data" and args.num_gpus > 1:
        eprint(
            f"Data-parallel mode: {args.num_gpus} independent single-GPU workers will process window shards. "
            "If the model does not fit on one GPU, switch to --parallelism context."
        )

    windows: list[WindowSpec] = []
    base = safe_name(input_video.stem)
    dummy_input_path = input_video

    for gen_i, desired_offset in enumerate(generation_offsets):
        plan = window_plans[desired_offset]
        start = plan.start_frame
        actual_cond_frames = plan.cond_frames
        actual_effective_cond_frames = effective_conditioning_frames(actual_cond_frames)
        actual_target_index = target_index_for_conditioning(actual_cond_frames, args.future_offset, args.target_index)
        actual_end = plan.end_frame
        sample_name = f"{base}_win_{gen_i:06d}_f{start:08d}_{actual_end:08d}_c{actual_cond_frames}"
        seed = args.seed + (gen_i if args.seed_per_window else 0)
        inference_type: Literal["image2world", "video2world"] = (
            "image2world" if actual_effective_cond_frames == 1 else "video2world"
        )

        if args.conditioning_mode == "clips":
            if inference_type == "image2world":
                input_path = clips_dir / f"{sample_name}.jpg"
                if not input_path.exists() or not args.resume:
                    ffmpeg_extract_image_frame(input_video, start, input_path)
            else:
                input_path = clips_dir / f"{sample_name}.mp4"
                if not input_path.exists() or not args.resume:
                    if actual_cond_frames == 5:
                        ffmpeg_extract_video_window(
                            input_video,
                            start,
                            actual_end,
                            input_path,
                            fps=fps,
                            codec=args.clip_codec,
                            crf=args.clip_crf,
                            preset=args.ffmpeg_preset,
                        )
                    else:
                        create_padded_five_frame_clip(
                            input_video=input_video,
                            start_frame=start,
                            cond_frames=actual_cond_frames,
                            output_video=input_path,
                            tmp_dir=work_dir / "pad_tmp" / sample_name,
                            fps=fps,
                            codec=args.clip_codec,
                            crf=args.clip_crf,
                            preset=args.ffmpeg_preset,
                        )
        else:
            # JSON still needs a valid input_path for pydantic validation. The direct
            # worker ignores this path and feeds a tensor into the official pipeline.
            input_path = dummy_input_path

        json_path = params_dir / f"{sample_name}.json"
        if not json_path.exists() or not args.resume:
            make_inference_json(
                json_path=json_path,
                sample_name=sample_name,
                inference_type=inference_type,
                input_path=input_path,
                prompt=prompt,
                num_output_frames=num_output_frames,
                seed=seed,
                guidance=args.guidance,
                num_steps=args.num_steps,
                resolution=args.resolution,
                negative_prompt=args.negative_prompt,
                enable_autoregressive=args.enable_autoregressive,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )

        windows.append(
            WindowSpec(
                index=gen_i,
                start_frame=start,
                end_frame=actual_end,
                output_start_index=plan.output_index,
                cond_frames=actual_cond_frames,
                effective_cond_frames=actual_effective_cond_frames,
                target_index=actual_target_index,
                input_path=str(input_path),
                json_path=str(json_path),
                sample_name=sample_name,
                inference_type=inference_type,
                seed=seed,
            )
        )

    setup: dict[str, Any] = {
        "output_dir": str(cosmos_output_dir),
        "model": args.model,
        "checkpoint_path": args.checkpoint_path,
        "experiment": args.experiment,
        "config_file": args.config_file or "",
        "context_parallel_size": args.num_gpus if parallelism == "context" else 1,
        "offload_diffusion_model": args.offload_diffusion_model,
        "offload_tokenizer": args.offload_tokenizer,
        "offload_text_encoder": args.offload_text_encoder,
        "disable_guardrails": args.disable_guardrails,
        "offload_guardrail_models": args.offload_guardrail_models,
        "keep_going": args.keep_going,
        "profile": args.profile,
        "skip_existing_output": False if args.direct_frame_output else args.resume,
    }

    manifest = Manifest(
        repo_root=str(repo_root),
        created_at_unix=time.time(),
        input_video=str(input_video),
        output_video=str(output_video),
        work_dir=str(work_dir),
        cosmos_output_dir=str(cosmos_output_dir),
        params_dir=str(params_dir),
        clips_dir=str(clips_dir),
        frames_dir=str(frames_dir),
        source_frames_dir=str(source_frames_dir),
        prompt=prompt,
        model=args.model,
        num_gpus=args.num_gpus,
        parallelism=parallelism,
        conditioning_mode=args.conditioning_mode,
        direct_frame_output=args.direct_frame_output,
        disable_guardrails_effective=args.disable_guardrails,
        fps=fps,
        output_fps=output_fps,
        window_schedule=args.window_schedule,
        cond_frames=args.cond_frames,
        effective_cond_frames=effective_cond_frames,
        stride=args.stride,
        future_offset=args.future_offset,
        num_output_frames=num_output_frames,
        native_num_output_frames=native_num_output_frames,
        runtime_state_t=runtime_state_t,
        temporal_compression_factor=args.temporal_compression_factor,
        target_index=target_index,
        max_target_index=max_target_index,
        samples_per_generation=args.samples_per_generation,
        desired_window_count=len(window_plans),
        aggregation=args.aggregation,
        resume=args.resume,
        cuda_empty_cache_every=args.cuda_empty_cache_every,
        use_tf32=args.tf32,
        memory_profile=args.memory_profile,
        prompt_cache=args.prompt_cache,
        prompt_cache_device=args.prompt_cache_device,
        prompt_cache_max_entries=args.prompt_cache_max_entries,
        clear_prompt_cache_every=args.clear_prompt_cache_every,
        cleanup_cpu_offload_after_each=args.cleanup_cpu_offload_after_each,
        gc_collect_every=args.gc_collect_every,
        cuda_ipc_collect=args.cuda_ipc_collect,
        recycle_worker_every=args.recycle_worker_every,
        cuda_alloc_conf=args.cuda_alloc_conf or "",
        windows=windows,
        setup=setup,
    )
    manifest_path = work_dir / "manifest.json"
    manifest.write(manifest_path)
    eprint(f"Wrote manifest: {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Worker speed patches and direct generation
# ---------------------------------------------------------------------------


def configure_torch_runtime(*, use_tf32: bool) -> None:
    import torch

    torch.enable_grad(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if use_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def install_fsdp_safe_text_encoder_patch() -> None:
    """Avoid moving FSDP-managed text encoders with Module.to().

    Cosmos-Predict2.5's TextEncoder.compute_text_embeddings_online calls
    ``self.model = self.model.to(self.device)`` before every embedding pass.
    That is fine for a normal module, but it can collide with FSDP/context
    parallel ownership. The runner only needs the text encoder to execute on
    its current managed device, so in distributed runs we temporarily make that
    one ``.to(self.device)`` call a no-op and leave all other behavior intact.
    """
    try:
        import torch
        from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder
    except Exception as exc:
        if rank0():
            eprint(f"Warning: could not install FSDP-safe text encoder patch: {exc}")
        return

    if getattr(TextEncoder, "_cosmos_sliding_fsdp_safe_patch", False):
        return

    original = TextEncoder.compute_text_embeddings_online

    def patched_compute_text_embeddings_online(self, *args, **kwargs):
        use_noop_to = bool(torch.distributed.is_available() and torch.distributed.is_initialized())
        model = getattr(self, "model", None)
        if not use_noop_to or model is None or not hasattr(model, "to"):
            return original(self, *args, **kwargs)

        original_to = model.to

        def noop_to(*to_args, **to_kwargs):
            if to_args == (getattr(self, "device", None),) and not to_kwargs:
                return model
            return original_to(*to_args, **to_kwargs)

        model.to = noop_to
        try:
            return original(self, *args, **kwargs)
        finally:
            model.to = original_to

    TextEncoder.compute_text_embeddings_online = patched_compute_text_embeddings_online
    TextEncoder._cosmos_sliding_fsdp_safe_patch = True
    if rank0():
        eprint("Installed FSDP-safe text encoder patch for sliding Video2World runner.")


def _tree_map_tensors(obj: Any, fn: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tree_map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_map_tensors(v, fn) for v in obj)
    return obj


def _detach_for_cache(obj: Any, *, device: str) -> Any:
    def fn(t):
        out = t.detach()
        if device == "cpu":
            out = out.to("cpu")
        return out.clone()

    return _tree_map_tensors(obj, fn)


def install_prompt_embedding_cache(
    pipe: Any,
    *,
    enabled: bool,
    max_entries: int,
    cache_device: Literal["cpu", "cuda"],
) -> dict[str, Any]:
    """Install a bounded text-embedding cache, or a no-op cache in low-VRAM mode.

    The previous Turbo build used an unbounded in-process cache. That is fast, but
    bad for long sliding jobs if captions vary or if embeddings remain on CUDA.
    This cache is bounded and can store tensors on CPU. Low-VRAM defaults disable it.
    """
    from collections import OrderedDict

    stats: dict[str, Any] = {"hits": 0, "misses": 0, "evictions": 0, "clears": 0, "enabled": bool(enabled)}
    if not enabled:
        stats["clear"] = lambda: None
        return stats
    if max_entries < 1:
        max_entries = 1

    def put(cache: OrderedDict[Any, Any], key: Any, value: Any) -> None:
        cache[key] = _detach_for_cache(value, device=cache_device)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
            stats["evictions"] += 1

    def get(cache: OrderedDict[Any, Any], key: Any) -> Any:
        cache.move_to_end(key)
        return cache[key]

    caches: list[OrderedDict[Any, Any]] = []

    text_encoder = getattr(getattr(pipe, "model", None), "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "compute_text_embeddings_online"):
        original = text_encoder.compute_text_embeddings_online
        cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        caches.append(cache)

        def cached_compute_text_embeddings_online(*, data_batch: dict[str, Any], input_caption_key: str):
            captions = tuple(str(x) for x in data_batch.get(input_caption_key, []))
            if captions in cache:
                stats["hits"] += 1
                return get(cache, captions)
            stats["misses"] += 1
            out = original(data_batch=data_batch, input_caption_key=input_caption_key)
            put(cache, captions, out)
            return out

        text_encoder.compute_text_embeddings_online = cached_compute_text_embeddings_online

        def clear() -> None:
            for c in caches:
                c.clear()
            stats["clears"] += 1
            gc.collect()

        stats["clear"] = clear
        return stats

    # Fallback path for checkouts that use precomputed embeddings via get_text_embedding.
    try:
        globals_dict = pipe._get_data_batch_input.__globals__
        original_get_text_embedding = globals_dict.get("get_text_embedding")
        if original_get_text_embedding is None:
            stats["clear"] = lambda: None
            return stats
        cache2: OrderedDict[str, Any] = OrderedDict()
        caches.append(cache2)

        def cached_get_text_embedding(prompt: str):
            if prompt in cache2:
                stats["hits"] += 1
                return get(cache2, prompt)
            stats["misses"] += 1
            out = original_get_text_embedding(prompt)
            put(cache2, prompt, out)
            return out

        globals_dict["get_text_embedding"] = cached_get_text_embedding
    except Exception:
        pass

    def clear() -> None:
        for c in caches:
            c.clear()
        stats["clears"] += 1
        gc.collect()

    stats["clear"] = clear
    return stats


def resolve_video_resolution(pipe: Any, resolution: str) -> tuple[int, int]:
    if resolution == "none":
        h, w = pipe.model.get_video_height_width()
        return int(h), int(w)
    parts = [int(x) for x in str(resolution).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Resolution must be 'H,W' or 'none', got: {resolution}")
    return parts[0], parts[1]


def source_frame_path(frames_dir: Path, frame_index: int) -> Path:
    # ffmpeg image2 starts numbering at 1: src_00000001 is source frame 0.
    for ext in ("png", "jpg", "jpeg"):
        p = frames_dir / f"src_{frame_index + 1:08d}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find cached source frame for index {frame_index} in {frames_dir}")


def build_conditioning_tensor_from_cache(
    *,
    manifest: Manifest,
    window: WindowSpec,
    model_required_frames: int,
    video_resolution: tuple[int, int],
    num_latent_conditional_frames: int,
) -> Any:
    import torch
    import torchvision
    from PIL import Image
    from cosmos_predict2._src.predict2.inference.video2world import resize_input

    frames_dir = Path(manifest.source_frames_dir)
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
    if frames_to_extract not in (1, 5):
        raise ValueError(f"Unsupported frames_to_extract={frames_to_extract}")

    raw_indices = [window.start_frame + i for i in range(window.cond_frames)]
    if frames_to_extract == 1:
        selected_indices = [raw_indices[-1]]
    else:
        if window.cond_frames >= 5:
            selected_indices = raw_indices[-5:]
        else:
            pad = 5 - window.cond_frames
            selected_indices = [raw_indices[0]] * pad + raw_indices

    frame_tensors = []
    for idx in selected_indices:
        img = Image.open(source_frame_path(frames_dir, idx)).convert("RGB")
        frame_tensors.append(torchvision.transforms.functional.to_tensor(img))  # C,H,W float [0,1]

    video_tchw = torch.stack(frame_tensors, dim=0)  # T,C,H,W float
    video_tchw = (video_tchw * 255.0).to(torch.uint8)
    video_tchw = resize_input(video_tchw, list(video_resolution))

    t, c, h, w = video_tchw.shape
    if t != frames_to_extract:
        raise RuntimeError(f"Internal error: got {t} conditioning frames, expected {frames_to_extract}")
    if model_required_frames < frames_to_extract:
        raise RuntimeError(
            f"model_required_frames={model_required_frames} is less than conditioning frames={frames_to_extract}"
        )

    full_cthw = torch.zeros(c, model_required_frames, h, w, dtype=torch.uint8)
    cond_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()
    full_cthw[:, :frames_to_extract, :, :] = cond_cthw
    if frames_to_extract < model_required_frames:
        last = cond_cthw[:, -1:, :, :]
        full_cthw[:, frames_to_extract:, :, :] = last.repeat(1, model_required_frames - frames_to_extract, 1, 1)
    return full_cthw.unsqueeze(0)  # B,C,T,H,W


def save_tensor_frame_png(video_01: Any, frame_index: int, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_index < 0 or frame_index >= int(video_01.shape[1]):
        raise IndexError(f"frame_index={frame_index} out of range for video with T={video_01.shape[1]}")
    frame = video_01[:, frame_index, :, :].clamp(0.0, 1.0)
    arr = (frame * 255.0).to(dtype=frame.dtype).clamp(0.0, 255.0).to("cpu")
    arr = arr.to(dtype=__import__("torch").uint8).permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(np.asarray(arr)).save(output_path)


def cuda_memory_snapshot(prefix: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available() or not rank0():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        eprint(f"{prefix}: cuda allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")
    except Exception:
        return


def release_python_and_cuda_caches(*, cuda_ipc_collect: bool, label: str = "") -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if cuda_ipc_collect:
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    finally:
        gc.collect()
    if label:
        cuda_memory_snapshot(label)


def force_offload_components_to_cpu(pipe: Any) -> None:
    """Put offload-managed submodules back on CPU after every window.

    The official pipeline already moves several components on and off GPU, but
    this function is intentionally defensive. It avoids leaving tokenizer/text
    encoder/DiT modules resident on CUDA after an exception or after rank0 frame
    extraction.
    """
    try:
        model = pipe.model
    except Exception:
        return
    try:
        if getattr(pipe, "offload_diffusion_model", False):
            if hasattr(model, "net") and model.net is not None:
                model.net = model.net.to("cpu")
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to("cpu")
        if getattr(pipe, "offload_tokenizer", False) and hasattr(model, "tokenizer"):
            tok = model.tokenizer
            if hasattr(tok, "encoder") and tok.encoder is not None:
                tok.encoder = tok.encoder.to("cpu")
            if hasattr(tok, "decoder") and tok.decoder is not None:
                tok.decoder = tok.decoder.to("cpu")
        if getattr(pipe, "offload_text_encoder", False) and getattr(model, "text_encoder", None) is not None:
            text_encoder = model.text_encoder
            if hasattr(text_encoder, "model") and text_encoder.model is not None:
                text_encoder.model = text_encoder.model.to("cpu")
    except Exception as exc:
        if rank0():
            eprint(f"Warning: defensive CPU offload failed: {exc}")


def direct_output_frames_exist(manifest: Manifest, window: WindowSpec) -> bool:
    frames_dir = Path(manifest.frames_dir)
    for j in range(manifest.samples_per_generation):
        out_idx = window.output_start_index + j
        if out_idx >= manifest.desired_window_count:
            break
        if not (frames_dir / f"frame_{out_idx:08d}.png").exists():
            return False
    return True


def run_direct_frame_generation(inference: Any, samples: list[Any], manifest: Manifest, cache_stats: dict[str, Any] | None = None) -> None:
    import torch
    import numpy as np
    from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
    from cosmos_predict2.config import path_to_str

    pipe = inference.pipe
    model_required_frames = pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)
    if rank0():
        eprint(f"Worker model_required_frames={model_required_frames}; manifest expects {manifest.num_output_frames}")
        cuda_memory_snapshot("After model load")

    frames_dir = Path(manifest.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, (sample, window) in enumerate(zip(samples, manifest.windows, strict=True)):
        video = None
        video_01 = None
        input_arg = None
        frames = None
        frames_np = None
        processed = None
        try:
            if manifest.resume and direct_output_frames_exist(manifest, window):
                if rank0():
                    eprint(f"[{i + 1}/{len(samples)}] skip existing target frames for {sample.name}")
                continue

            if rank0():
                eprint(f"[{i + 1}/{len(samples)}] generating {sample.name}")
                cuda_memory_snapshot("Before generation")

            # Text guardrail only. Video guardrail is usually disabled for speed; if it is
            # enabled, we run it before extracting frames to preserve official behavior.
            if getattr(inference, "text_guardrail_runner", None) is not None:
                if not guardrail_presets.run_text_guardrail(sample.prompt, inference.text_guardrail_runner):
                    message = f"Guardrail blocked prompt for sample {sample.name}"
                    if inference.setup_args.keep_going:
                        eprint(message)
                        continue
                    raise RuntimeError(message)

            if manifest.conditioning_mode == "tensor":
                resolution = resolve_video_resolution(pipe, sample.resolution)
                input_arg = build_conditioning_tensor_from_cache(
                    manifest=manifest,
                    window=window,
                    model_required_frames=model_required_frames,
                    video_resolution=resolution,
                    num_latent_conditional_frames=sample.num_input_frames,
                )
            else:
                input_arg = path_to_str(sample.input_path)

            with torch.inference_mode():
                if sample.enable_autoregressive:
                    video = pipe.generate_autoregressive_from_batch(
                        prompt=sample.prompt,
                        input_path=input_arg,
                        num_output_frames=sample.num_output_frames,
                        chunk_size=sample.chunk_size,
                        chunk_overlap=sample.chunk_overlap,
                        guidance=sample.guidance,
                        num_latent_conditional_frames=sample.num_input_frames,
                        resolution=sample.resolution,
                        seed=sample.seed,
                        negative_prompt=sample.negative_prompt,
                        num_steps=sample.num_steps,
                    )
                else:
                    video = pipe.generate_vid2world(
                        prompt=sample.prompt,
                        input_path=input_arg,
                        guidance=sample.guidance,
                        num_video_frames=sample.num_output_frames,
                        num_latent_conditional_frames=sample.num_input_frames,
                        resolution=sample.resolution,
                        seed=sample.seed,
                        negative_prompt=sample.negative_prompt,
                        num_steps=sample.num_steps,
                    )

            if rank0():
                video_01 = (1.0 + video[0]) / 2.0  # C,T,H,W
                if getattr(inference, "video_guardrail_runner", None) is not None:
                    frames = (video_01 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames_np = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                    processed = guardrail_presets.run_video_guardrail(frames_np, inference.video_guardrail_runner)
                    if processed is None:
                        message = f"Guardrail blocked generated video for sample {sample.name}"
                        if inference.setup_args.keep_going:
                            eprint(message)
                            continue
                        raise RuntimeError(message)
                    video_01 = torch.from_numpy(processed).float().permute(3, 0, 1, 2) / 255.0

                for j in range(manifest.samples_per_generation):
                    out_idx = window.output_start_index + j
                    if out_idx >= manifest.desired_window_count:
                        break
                    target_index = window.target_index + j * manifest.stride
                    if target_index >= video_01.shape[1]:
                        raise RuntimeError(
                            f"Generated video for {sample.name} has only {video_01.shape[1]} frames; "
                            f"cannot extract target_index={target_index}"
                        )
                    out_path = frames_dir / f"frame_{out_idx:08d}.png"
                    save_tensor_frame_png(video_01, target_index, out_path)

        finally:
            # Drop all per-window references before trimming the CUDA allocator.
            del processed, frames_np, frames, video_01, video, input_arg
            if manifest.cleanup_cpu_offload_after_each:
                force_offload_components_to_cpu(pipe)
            if manifest.gc_collect_every > 0 and (i + 1) % manifest.gc_collect_every == 0:
                gc.collect()
            if (
                cache_stats is not None
                and manifest.clear_prompt_cache_every > 0
                and (i + 1) % manifest.clear_prompt_cache_every == 0
            ):
                clear_fn = cache_stats.get("clear")
                if callable(clear_fn):
                    clear_fn()
            if manifest.cuda_empty_cache_every > 0 and (i + 1) % manifest.cuda_empty_cache_every == 0:
                release_python_and_cuda_caches(
                    cuda_ipc_collect=manifest.cuda_ipc_collect,
                    label=f"After cleanup {i + 1}/{len(samples)}" if rank0() else "",
                )


def run_official_cosmos_worker(manifest_path: Path) -> None:
    manifest = Manifest.from_path(manifest_path)
    repo_root = Path(manifest.repo_root).resolve()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    configure_torch_runtime(use_tf32=manifest.use_tf32)
    install_fsdp_safe_text_encoder_patch()

    from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
    from cosmos_predict2.config import InferenceArguments, SetupArguments

    init_environment()
    try:
        setup_data = {k: v for k, v in manifest.setup.items() if v is not None}
        setup_args = SetupArguments.model_validate(setup_data)
        input_files = [Path(w.json_path) for w in manifest.windows]
        inference_samples = InferenceArguments.from_files(input_files, overrides=None, setup_args=setup_args)
        init_output_dir(setup_args.output_dir, profile=setup_args.profile)

        from cosmos_predict2.inference import Inference

        inference = Inference(setup_args)
        if not hasattr(inference.pipe.model.config, "state_t"):
            raise RuntimeError("Loaded Cosmos model config has no state_t attribute; cannot shrink runtime frames.")
        old_state_t = int(inference.pipe.model.config.state_t)
        if old_state_t != manifest.runtime_state_t:
            if rank0():
                eprint(
                    f"Overriding model.config.state_t: {old_state_t} -> {manifest.runtime_state_t} "
                    f"for {manifest.num_output_frames} pixel frames"
                )
            inference.pipe.model.config.state_t = manifest.runtime_state_t

        cache_stats = install_prompt_embedding_cache(
            inference.pipe,
            enabled=manifest.prompt_cache,
            max_entries=manifest.prompt_cache_max_entries,
            cache_device=manifest.prompt_cache_device,
        )
        try:
            if manifest.direct_frame_output:
                run_direct_frame_generation(inference, inference_samples, manifest, cache_stats=cache_stats)
            else:
                output_paths = inference.generate(inference_samples, output_dir=setup_args.output_dir)
                if rank0():
                    eprint(f"Cosmos worker completed. Generated {len(output_paths)} files.")
        finally:
            clear_fn = cache_stats.get("clear")
            if callable(clear_fn):
                clear_fn()
            try:
                force_offload_components_to_cpu(inference.pipe)
            except Exception:
                pass
            release_python_and_cuda_caches(cuda_ipc_collect=manifest.cuda_ipc_collect, label="Worker final cleanup" if rank0() else "")

        if rank0():
            eprint(
                "Prompt embedding cache stats: "
                f"enabled={cache_stats.get('enabled')}, hits={cache_stats.get('hits')}, "
                f"misses={cache_stats.get('misses')}, evictions={cache_stats.get('evictions')}, "
                f"clears={cache_stats.get('clears')}"
            )
    finally:
        cleanup_environment()


# ---------------------------------------------------------------------------
# Launch and assembly
# ---------------------------------------------------------------------------


def generated_path_for_window(manifest: Manifest, w: WindowSpec) -> Path:
    return Path(manifest.cosmos_output_dir) / f"{w.sample_name}.mp4"


def split_windows_for_recycling(windows: list[WindowSpec], recycle_every: int) -> list[list[WindowSpec]]:
    if recycle_every <= 0 or len(windows) <= recycle_every:
        return [windows]
    return [windows[i : i + recycle_every] for i in range(0, len(windows), recycle_every)]


def write_recycle_manifest(base: Manifest, windows: list[WindowSpec], batch_dir: Path, *, batch_name: str) -> Path:
    batch_setup = dict(base.setup)
    batch_output_dir = Path(base.cosmos_output_dir) / "recycle_outputs" / batch_name
    batch_setup["output_dir"] = str(batch_output_dir)
    batch_manifest = replace(
        base,
        cosmos_output_dir=str(batch_output_dir),
        setup=batch_setup,
        windows=windows,
    )
    path = batch_dir / batch_name / "manifest.json"
    batch_manifest.write(path)
    return path


def launch_context_worker_recycled(manifest_path: Path, *, num_gpus: int, repo_root: Path) -> None:
    manifest = Manifest.from_path(manifest_path)
    batches = split_windows_for_recycling(manifest.windows, manifest.recycle_worker_every)
    if len(batches) <= 1:
        launch_context_worker(manifest_path, num_gpus=num_gpus, repo_root=repo_root)
        return
    batch_root = Path(manifest.work_dir) / "recycle_context"
    eprint(f"Worker recycling enabled: {len(batches)} batches of up to {manifest.recycle_worker_every} generations")
    for bi, windows in enumerate(batches):
        bp = write_recycle_manifest(manifest, windows, batch_root, batch_name=f"batch_{bi:04d}")
        eprint(f"Recycle batch {bi + 1}/{len(batches)}: {len(windows)} generations")
        launch_context_worker(bp, num_gpus=num_gpus, repo_root=repo_root)


def launch_context_worker(manifest_path: Path, *, num_gpus: int, repo_root: Path) -> None:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
    env["NUM_GPUS"] = str(num_gpus)
    try:
        m = Manifest.from_path(manifest_path)
        if m.cuda_alloc_conf:
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", m.cuda_alloc_conf)
    except Exception:
        pass

    if num_gpus > 1:
        require_executable("torchrun")
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(script_path),
            "--_worker-manifest",
            str(manifest_path),
        ]
    else:
        cmd = [sys.executable, str(script_path), "--_worker-manifest", str(manifest_path)]
    run_cmd(cmd, cwd=repo_root, env=env)


def launch_data_workers(manifest_path: Path, *, gpu_ids: list[str], repo_root: Path) -> None:
    manifest = Manifest.from_path(manifest_path)
    if not manifest.direct_frame_output:
        raise RuntimeError("Data-parallel launch requires direct_frame_output=True")

    chunks = chunk_contiguous(manifest.windows, len(gpu_ids))
    script_path = Path(__file__).resolve()

    shard_batch_lists: list[list[list[WindowSpec]]] = []
    for windows in chunks:
        shard_batch_lists.append(split_windows_for_recycling(windows, manifest.recycle_worker_every))
    max_rounds = max((len(x) for x in shard_batch_lists), default=0)

    for round_idx in range(max_rounds):
        procs: list[tuple[int, str, subprocess.Popen[None]]] = []
        for shard_idx, gpu_id in enumerate(gpu_ids):
            if round_idx >= len(shard_batch_lists[shard_idx]):
                continue
            windows = shard_batch_lists[shard_idx][round_idx]
            if not windows:
                continue
            shard_dir = Path(manifest.work_dir) / "shards" / f"shard_{shard_idx:02d}" / f"batch_{round_idx:04d}"
            shard_manifest_path = shard_dir / "manifest.json"
            shard_setup = dict(manifest.setup)
            shard_output_dir = Path(manifest.cosmos_output_dir) / f"shard_{shard_idx:02d}" / f"batch_{round_idx:04d}"
            shard_setup["output_dir"] = str(shard_output_dir)
            shard_setup["context_parallel_size"] = 1
            shard_manifest = replace(
                manifest,
                num_gpus=1,
                parallelism="context",
                cosmos_output_dir=str(shard_output_dir),
                setup=shard_setup,
                windows=windows,
            )
            shard_manifest.write(shard_manifest_path)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.pop("WORLD_SIZE", None)
            env.pop("RANK", None)
            env.pop("LOCAL_RANK", None)
            env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
            env["NUM_GPUS"] = "1"
            if manifest.cuda_alloc_conf:
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", manifest.cuda_alloc_conf)
            env["COSMOS_TURBO_SHARD"] = str(shard_idx)
            cmd = [sys.executable, str(script_path), "--_worker-manifest", str(shard_manifest_path)]
            eprint(
                f"Launching data shard {shard_idx} round {round_idx + 1}/{max_rounds} "
                f"on CUDA_VISIBLE_DEVICES={gpu_id}: {len(windows)} generations"
            )
            eprint(f"$ {shlex_join(cmd)}")
            procs.append((shard_idx, gpu_id, subprocess.Popen(cmd, cwd=str(repo_root), env=env)))

        failures: list[tuple[int, str, int]] = []
        try:
            for shard_idx, gpu_id, proc in procs:
                code = proc.wait()
                if code != 0:
                    failures.append((shard_idx, gpu_id, code))
        finally:
            if failures:
                for _, _, proc in procs:
                    if proc.poll() is None:
                        proc.terminate()

        if failures:
            details = ", ".join(f"shard={s}/gpu={g}/exit={c}" for s, g, c in failures)
            raise RuntimeError(f"Data-parallel worker failure(s): {details}")



def assemble_direct_frames(manifest: Manifest, *, codec: str, crf: int, preset: str) -> None:
    frames_dir = Path(manifest.frames_dir)
    missing = [i for i in range(manifest.desired_window_count) if not (frames_dir / f"frame_{i:08d}.png").exists()]
    if missing:
        preview = ", ".join(str(i) for i in missing[:20])
        raise RuntimeError(f"Missing {len(missing)} assembled frames in {frames_dir}; first missing: {preview}")
    ffmpeg_concat_images_to_video(
        frames_dir,
        Path(manifest.output_video),
        fps=manifest.output_fps,
        codec=codec,
        crf=crf,
        preset=preset,
    )
    eprint(f"Assembled final video: {manifest.output_video} ({manifest.desired_window_count} frames)")


def assemble_target_frame_video_from_mp4(manifest: Manifest, *, on_short: str, codec: str, crf: int, preset: str) -> None:
    frames_dir = Path(manifest.frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for w in manifest.windows:
        gen_path = generated_path_for_window(manifest, w)
        if not gen_path.exists():
            raise RuntimeError(f"Missing generated output for {w.sample_name}: {gen_path}")
        info = ffprobe_video(gen_path)
        for j in range(manifest.samples_per_generation):
            out_idx = w.output_start_index + j
            if out_idx >= manifest.desired_window_count:
                break
            target_index = w.target_index + j * manifest.stride
            if target_index >= info.frame_count:
                if on_short == "last":
                    target_index = info.frame_count - 1
                elif on_short == "skip":
                    eprint(f"Skipping {gen_path}: target_index={target_index}, frame_count={info.frame_count}")
                    continue
                else:
                    raise RuntimeError(f"{gen_path} is too short: target_index={target_index}, frame_count={info.frame_count}")
            frame_path = frames_dir / f"frame_{out_idx:08d}.png"
            ffmpeg_extract_image_frame(gen_path, target_index, frame_path)
            extracted += 1

    if extracted == 0:
        raise RuntimeError("No frames were extracted; cannot assemble output video")
    assemble_direct_frames(manifest, codec=codec, crf=crf, preset=preset)


def assemble_generated_tail_video(manifest: Manifest, *, on_short: str, codec: str, crf: int, preset: str) -> None:
    frames_dir = Path(manifest.frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for w in manifest.windows:
        start_idx = w.effective_cond_frames
        end_idx = w.target_index
        gen_path = generated_path_for_window(manifest, w)
        if not gen_path.exists():
            raise RuntimeError(f"Missing generated output for {w.sample_name}: {gen_path}")
        info = ffprobe_video(gen_path)
        if start_idx >= info.frame_count:
            if on_short == "skip":
                continue
            if on_short == "last":
                local_start = info.frame_count - 1
                local_end = info.frame_count - 1
            else:
                raise RuntimeError(f"{gen_path} is too short: {info.frame_count} frames")
        else:
            local_start = start_idx
            local_end = min(end_idx, info.frame_count - 1)
        for idx in range(local_start, local_end + 1):
            frame_path = frames_dir / f"frame_{extracted:08d}.png"
            ffmpeg_extract_image_frame(gen_path, idx, frame_path)
            extracted += 1

    if extracted == 0:
        raise RuntimeError("No frames were extracted; cannot assemble output video")
    ffmpeg_concat_images_to_video(frames_dir, Path(manifest.output_video), fps=manifest.output_fps, codec=codec, crf=crf, preset=preset)
    eprint(f"Assembled generated-tail video: {manifest.output_video} ({extracted} frames)")


def assemble_output(manifest_path: Path, *, on_short: str, codec: str, crf: int, preset: str) -> None:
    manifest = Manifest.from_path(manifest_path)
    if manifest.direct_frame_output:
        assemble_direct_frames(manifest, codec=codec, crf=crf, preset=preset)
    elif manifest.aggregation == "target_frame":
        assemble_target_frame_video_from_mp4(manifest, on_short=on_short, codec=codec, crf=crf, preset=preset)
    elif manifest.aggregation == "generated_tail":
        assemble_generated_tail_video(manifest, on_short=on_short, codec=codec, crf=crf, preset=preset)
    else:
        raise ValueError(f"Unknown aggregation: {manifest.aggregation}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turbo sliding-window Cosmos-Predict2.5 Video2World driver.",
        epilog=textwrap.dedent(
            """
            Profiles:
              exact     : exact sliding-window semantics, 35 steps, 1 target frame per generation.
              fast      : approximate, 28 steps, 2 target frames per generation.
              veryfast  : approximate, 20 steps, 4 target frames per generation.
              insane    : approximate, 12 steps, 8 target frames per generation.

            Do not run this wrapper with torchrun. Use --num-gpus N.
            """
        ),
    )

    p.add_argument("--_worker-manifest", default=None, help=argparse.SUPPRESS)

    # Main IO.
    p.add_argument("--repo-root", default=".", help="Cosmos-Predict2.5 repo root. Default: current directory.")
    p.add_argument("--input-video", help="Source video used for sliding-window conditioning.")
    p.add_argument("--output-video", help="Final assembled mp4 path.")
    p.add_argument("--work-dir", default=None, help="Working directory. Default: <output>_work.")
    p.add_argument("--overwrite-workdir", action="store_true", help="Delete work-dir before preparing.")
    p.add_argument("--resume", action="store_true", help="Reuse source cache / generated frames where possible.")
    p.add_argument("--cleanup", action="store_true", help="Delete work-dir after successful assembly.")
    p.add_argument("--dry-run", action="store_true", help="Prepare source cache/json/manifest only; do not run Cosmos.")

    # Prompt.
    p.add_argument("--prompt", default=None, help="Prompt for all windows.")
    p.add_argument("--prompt-file", default=None, help="Text file containing the prompt.")
    p.add_argument("--negative-prompt", default=None, help="Optional negative prompt override.")

    # Strategy.
    p.add_argument("--turbo-profile", choices=["exact", "fast", "veryfast", "insane"], default="exact")
    p.add_argument("--memory-profile", choices=["speed", "balanced", "lowvram", "ultralowvram"], default="speed", help="Default: speed for H200-class servers.")
    p.add_argument("--parallelism", choices=["auto", "context", "data"], default="auto", help="auto: lowvram=>context; speed/balanced 2B+multiGPU=>data.")
    p.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids for data parallel mode. Default: 0..num_gpus-1.")
    p.add_argument("--conditioning-mode", choices=["tensor", "clips"], default="tensor", help="tensor avoids per-window mp4 files. Default: tensor.")
    p.add_argument("--direct-frame-output", action=argparse.BooleanOptionalAction, default=True, help="Save target frames directly from tensors. Default: true.")
    p.add_argument("--samples-per-generation", type=int, default=None, help="Override profile target frames extracted per Cosmos generation.")

    # Sliding window behavior.
    p.add_argument("--window-schedule", choices=["warmup5", "fixed"], default="warmup5", help="warmup5: 1, 1-2, ..., 1-5, 2-6... Default: warmup5.")
    p.add_argument("--cond-frames", type=int, default=5, help="Maximum conditioning frames. warmup5 supports 1..5; fixed official-clean values are 1 or 5. Default: 5.")
    p.add_argument("--allow-pad-cond-frames", action="store_true", help="Allow fixed cond_frames=2..4 by left-padding into 5 frames. warmup5 pads 2..4 automatically.")
    p.add_argument("--stride", type=int, default=1, help="Source-frame shift between desired windows. Default: 1.")
    p.add_argument("--future-offset", type=int, default=9, help="N-th future frame after the last real conditioning frame. Use 9 for source frame 1 -> absolute frame 10. Default: 9.")
    p.add_argument("--target-index", type=int, default=None, help="Generated-clip frame index override. Default: effective_cond_frames + future_offset - 1 per window.")
    p.add_argument("--start-frame", type=int, default=0, help="First source frame index. Default: 0.")
    p.add_argument("--stop-frame", type=int, default=None, help="Exclusive source stop frame. Default: end of video.")
    p.add_argument("--max-windows", type=int, default=None, help="Limit desired windows for testing.")
    p.add_argument("--aggregation", choices=["target_frame", "generated_tail"], default="target_frame")
    p.add_argument("--on-short-output", choices=["error", "last", "skip"], default="error")

    # Video/framerate/encoding.
    p.add_argument("--fps", type=float, default=16.0, help="FPS for temporary conditioning clips / final default. Default: 16.")
    p.add_argument("--output-fps", type=float, default=None, help="FPS for final assembled video. Default: --fps.")
    p.add_argument("--source-frame-ext", choices=["png", "jpg"], default="png", help="Source frame cache format for tensor mode. Default: png.")
    p.add_argument("--source-jpg-quality", type=int, default=2, help="JPEG quality if --source-frame-ext jpg. Lower is better. Default: 2.")
    p.add_argument("--clip-codec", default="libx264", help="ffmpeg codec for temporary clips/final output. Default: libx264.")
    p.add_argument("--clip-crf", type=int, default=16, help="CRF for temporary clips/final output. Default: 16.")
    p.add_argument("--ffmpeg-preset", default="veryfast", help="ffmpeg preset. Default: veryfast.")

    # Cosmos setup args.
    p.add_argument("--model", default="14B/post-trained", help="2B/post-trained or 14B/post-trained. Default: 14B/post-trained.")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--config-file", default="")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--offload-diffusion-model", action=argparse.BooleanOptionalAction, default=None, help="Profile default. Use --no-offload-diffusion-model to force resident DiT.")
    p.add_argument("--offload-tokenizer", action=argparse.BooleanOptionalAction, default=None, help="Profile default. Use --no-offload-tokenizer to keep tokenizer on GPU.")
    p.add_argument("--offload-text-encoder", action=argparse.BooleanOptionalAction, default=None, help="Profile default. Use --no-offload-text-encoder to keep text encoder on GPU.")
    p.add_argument("--disable-guardrails", action=argparse.BooleanOptionalAction, default=True, help="Default true for speed/VRAM. Use --no-disable-guardrails to restore guardrails.")
    p.add_argument("--offload-guardrail-models", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile", action="store_true")

    # Cosmos inference args.
    p.add_argument("--frame-budget", choices=["auto", "native"], default=None, help="Default: auto.")
    p.add_argument("--num-output-frames", type=int, default=None, help="Minimum pixel frames before tokenizer rounding.")
    p.add_argument("--runtime-state-t", type=int, default=None, help="Expert override for model.config.state_t.")
    p.add_argument("--temporal-compression-factor", type=int, default=4)
    p.add_argument("--align-latent-to-parallel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-steps", type=int, default=None, help="Profile default unless explicitly set.")
    p.add_argument("--guidance", type=int, default=7)
    p.add_argument("--resolution", default="none", help='Resolution string, e.g. "none" or "704,1280".')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-per-window", action="store_true")
    p.add_argument("--enable-autoregressive", action="store_true")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-overlap", type=int, default=None)

    # Runtime behavior.
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 runtime flags where applicable. Default: true.")
    p.add_argument("--cuda-empty-cache-every", type=int, default=None, help="Call torch.cuda.empty_cache every N generations. Profile default.")
    p.add_argument("--gc-collect-every", type=int, default=None, help="Call Python gc.collect every N generations. Profile default.")
    p.add_argument("--cuda-ipc-collect", action=argparse.BooleanOptionalAction, default=None, help="Also call torch.cuda.ipc_collect during cleanup. Profile default.")
    p.add_argument("--cleanup-cpu-offload-after-each", action=argparse.BooleanOptionalAction, default=None, help="Defensively move offload-managed modules back to CPU after each generation. Profile default.")
    p.add_argument("--prompt-cache", action=argparse.BooleanOptionalAction, default=None, help="Enable bounded prompt embedding cache. Profile default.")
    p.add_argument("--prompt-cache-device", choices=["cpu", "cuda"], default=None, help="Where cached text embeddings live. Profile default.")
    p.add_argument("--prompt-cache-max-entries", type=int, default=None, help="Maximum prompt-cache entries. Profile default.")
    p.add_argument("--clear-prompt-cache-every", type=int, default=None, help="Clear prompt cache every N generations if cache is enabled. 0 disables. Profile default.")
    p.add_argument("--recycle-worker-every", type=int, default=None, help="Restart worker process after N generations. 0 disables. Profile default.")
    p.add_argument("--cuda-alloc-conf", default=None, help="Value for PYTORCH_CUDA_ALLOC_CONF in worker processes. Profile default.")

    return p


def validate_parent_args(args: argparse.Namespace) -> None:
    if args._worker_manifest:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if rank0():
            eprint("Do not launch this wrapper directly with torchrun. Run it with --num-gpus N.")
        raise SystemExit(2)
    if not args.input_video:
        raise SystemExit("--input-video is required")
    if not args.output_video:
        raise SystemExit("--output-video is required")
    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")
    if args.guidance < 0 or args.guidance > 7:
        raise SystemExit("--guidance must be in [0, 7]")
    if args.temporal_compression_factor < 1:
        raise SystemExit("--temporal-compression-factor must be >= 1")
    if args.runtime_state_t is not None and args.runtime_state_t < 1:
        raise SystemExit("--runtime-state-t must be >= 1")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args._worker_manifest:
        run_official_cosmos_worker(Path(args._worker_manifest).expanduser().resolve())
        return

    apply_turbo_profile(args)
    apply_memory_profile(args)
    validate_parent_args(args)
    manifest_path = prepare_windows_and_manifest(args)
    if args.dry_run:
        eprint("Dry run complete. Cosmos was not launched.")
        return

    manifest = Manifest.from_path(manifest_path)
    repo_root = Path(manifest.repo_root)
    if manifest.parallelism == "data" and manifest.num_gpus > 1:
        gpu_ids = parse_gpu_ids(args.gpu_ids, manifest.num_gpus)
        if len(gpu_ids) != manifest.num_gpus:
            raise SystemExit(f"--gpu-ids count ({len(gpu_ids)}) must match --num-gpus ({manifest.num_gpus})")
        launch_data_workers(manifest_path, gpu_ids=gpu_ids, repo_root=repo_root)
    else:
        launch_context_worker_recycled(manifest_path, num_gpus=manifest.num_gpus, repo_root=repo_root)

    assemble_output(manifest_path, on_short=args.on_short_output, codec=args.clip_codec, crf=args.clip_crf, preset=args.ffmpeg_preset)

    if args.cleanup:
        shutil.rmtree(Path(manifest.work_dir), ignore_errors=True)
        eprint(f"Cleaned work dir: {manifest.work_dir}")


if __name__ == "__main__":
    main()
