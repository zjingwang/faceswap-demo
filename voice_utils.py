#!/usr/bin/env python3
"""Local voice extract / best-effort replacement / ffmpeg mux (no CUDA).

Full RVC/TTS cloning is too heavy for Radeon 780M shared RAM.
When a source voice clip is provided:
  - normalize to 16k mono
  - loop/pad to match video duration
  - mux onto the face-swapped (silent) video
Otherwise keep / re-mux the original video audio when present.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Callable, Optional

LogFn = Callable[[str], None]

_FFMPEG_INSTALL_HINT = (
    "未找到 ffmpeg，请安装并加入 PATH（Windows: winget install Gyan.FFmpeg），"
    "或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe"
)


def _candidate_bin_dirs() -> list[str]:
    """Known install locations when PATH is stale (e.g. Gradio launched before winget)."""
    dirs: list[str] = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        winget = os.path.join(local, "Microsoft", "WinGet", "Packages")
        pattern = os.path.join(
            winget, "Gyan.FFmpeg*", "ffmpeg-*-full_build", "bin"
        )
        dirs.extend(sorted(glob.glob(pattern), reverse=True))
    for base in (
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ffmpeg\bin",
        os.path.expanduser(r"~\ffmpeg\bin"),
    ):
        if base not in dirs:
            dirs.append(base)
    return dirs


def _resolve_tool(name: str, env_key: str) -> Optional[str]:
    """Resolve ffmpeg/ffprobe: env override -> PATH -> known Windows locations."""
    override = (os.environ.get(env_key) or "").strip().strip('"')
    if override:
        if os.path.isdir(override):
            candidate = os.path.join(override, f"{name}.exe" if os.name == "nt" else name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        if os.path.isfile(override):
            # Allow FFMPEG_PATH=.../ffmpeg.exe to also locate sibling ffprobe
            if name == "ffmpeg" or os.path.basename(override).lower().startswith(name):
                return os.path.abspath(override)
            sibling = os.path.join(os.path.dirname(override), f"{name}.exe" if os.name == "nt" else name)
            if os.path.isfile(sibling):
                return os.path.abspath(sibling)

    path = shutil.which(name)
    if path:
        return os.path.abspath(path)

    exe = f"{name}.exe" if os.name == "nt" else name
    for d in _candidate_bin_dirs():
        candidate = os.path.join(d, exe)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str:
    path = _resolve_tool("ffmpeg", "FFMPEG_PATH")
    if not path:
        raise RuntimeError(_FFMPEG_INSTALL_HINT)
    return path


@lru_cache(maxsize=1)
def resolve_ffprobe() -> Optional[str]:
    path = _resolve_tool("ffprobe", "FFPROBE_PATH")
    if path:
        return path
    # Sibling of resolved ffmpeg (common when FFMPEG_PATH points at ffmpeg.exe)
    try:
        ffmpeg = resolve_ffmpeg()
    except RuntimeError:
        return None
    sibling = os.path.join(
        os.path.dirname(ffmpeg),
        "ffprobe.exe" if os.name == "nt" else "ffprobe",
    )
    if os.path.isfile(sibling):
        return os.path.abspath(sibling)
    return None


def _which_ffmpeg() -> str:
    return resolve_ffmpeg()


def run_ffmpeg(args: list, log: Optional[LogFn] = None) -> None:
    ffmpeg = resolve_ffmpeg()
    cmd = [ffmpeg, "-y", *args]
    if log:
        log(f"[ffmpeg] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"ffmpeg 失败 (code={proc.returncode}): {err}")


def probe_duration_seconds(media_path: str) -> float:
    """Best-effort duration via ffprobe; 0 if unknown."""
    ffprobe = resolve_ffprobe()
    if not ffprobe:
        return 0.0
    proc = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return float((proc.stdout or "").strip())
    except ValueError:
        return 0.0


def extract_audio(video_path: str, wav_path: str, log: Optional[LogFn] = None) -> bool:
    if log:
        log("正在从目标视频提取音频…")
    try:
        run_ffmpeg(
            [
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", wav_path,
            ],
            log=log,
        )
        return os.path.isfile(wav_path) and os.path.getsize(wav_path) > 44
    except RuntimeError as e:
        if log:
            log(f"提取音频失败或视频无音轨: {e}")
        return False


def replace_with_source_voice(
    source_voice_path: str,
    duration_sec: float,
    out_wav: str,
    log: Optional[LogFn] = None,
) -> str:
    """Loop/pad source clip to duration_sec. Not neural voice cloning."""
    if log:
        log(
            "人声路径=最佳努力替换：将源声音片段循环/填充到视频时长 "
            "（非 RVC/神经网络克隆；本地 ffmpeg，无 CUDA）。"
        )
    if duration_sec <= 0:
        duration_sec = 10.0
        if log:
            log(f"无法探测时长，回退为 {duration_sec:.1f}s")
    work = tempfile.mkdtemp(prefix="voice_norm_")
    try:
        norm = os.path.join(work, "norm.wav")
        run_ffmpeg(
            ["-i", source_voice_path, "-ac", "1", "-ar", "16000", norm],
            log=log,
        )
        run_ffmpeg(
            [
                "-stream_loop", "-1",
                "-i", norm,
                "-t", f"{duration_sec:.3f}",
                "-ac", "1",
                "-ar", "16000",
                out_wav,
            ],
            log=log,
        )
        if log:
            log(f"源声音已对齐到 {duration_sec:.2f}s -> {out_wav}")
        return out_wav
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process_voice(
    target_wav: str,
    source_voice_path: Optional[str],
    out_wav: str,
    *,
    video_path: Optional[str] = None,
    log: Optional[LogFn] = None,
) -> str:
    """Build final wav: prefer replace-with-source; else copy target audio."""
    duration = 0.0
    if video_path:
        duration = probe_duration_seconds(video_path)
    if duration <= 0 and target_wav and os.path.isfile(target_wav):
        duration = probe_duration_seconds(target_wav)

    if source_voice_path and os.path.isfile(source_voice_path):
        return replace_with_source_voice(source_voice_path, duration, out_wav, log=log)

    if log:
        log("未提供源声音，将使用目标视频原音轨（若有）。")
    if target_wav and os.path.isfile(target_wav):
        shutil.copy2(target_wav, out_wav)
        return out_wav
    raise RuntimeError("无源声音且目标视频无音轨，无法生成音频。")


def mux_av(
    video_no_audio: str,
    audio_wav: str,
    output_path: str,
    log: Optional[LogFn] = None,
) -> str:
    if log:
        log("正在音视频封装（mux）…")
    run_ffmpeg(
        [
            "-i", video_no_audio,
            "-i", audio_wav,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ],
        log=log,
    )
    if log:
        log(f"封装完成 -> {output_path}")
    return output_path


def make_work_dir(prefix: str = "faceswap_") -> str:
    return tempfile.mkdtemp(prefix=prefix)
