import csv
import functools
import json
import logging
import os
import shutil
import stat
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, List, Optional

import gradio as gr
import torch
from PIL import Image

import auth
import cleanup as cleanup_jobs
import task_store
from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)

MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODEL_DIR / "outputs"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch"
SINGLE_OUTPUT_DIR = OUTPUT_DIR / "single"
USERS_OUTPUT_DIR = OUTPUT_DIR / "users"
LOCAL_BATCH_INPUT_DIR = MODEL_DIR / "batch_inputs"
RETENTION_DAYS = int(os.getenv("OUTPUT_RETENTION_DAYS", "7"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(6 * 60 * 60)))
METADATA_FILENAME = "metadata.json"
IN_PROGRESS_MARKER = ".in_progress"
COMPLETED_MARKER = ".completed"
FAILED_MARKER = ".failed"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USERS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)
BATCH_MODE_REMOTE = "远程上传模式（manifest + ZIP）"
BATCH_MODE_LOCAL = "服务端本地图片模式"
BATCH_STATUS_PENDING = "pending"
BATCH_STATUS_RUNNING = "running"
BATCH_STATUS_FINALIZING = "finalizing"
BATCH_STATUS_COMPLETED = "completed"
BATCH_STATUS_FAILED = "failed"
PARTIAL_RESULTS_FILENAME = "batch_results.partial.json"
ACTIVE_BATCH_THREADS: dict[str, threading.Thread] = {}
ACTIVE_BATCH_THREADS_LOCK = threading.Lock()

_PIPELINE: Optional[QwenImageEditPlusPipeline] = None
_DEVICE: Optional[str] = None
_DTYPE: Optional[torch.dtype] = None
_DEVICE_MAP_INFO: Optional[str] = None
_PIPELINE_EXECUTION_DEVICE: Optional[str] = None


@dataclass
class BatchItem:
    row_id: str
    prompt: str
    negative_prompt: str
    image_refs: List[str]
    image_paths: List[Path]
    seed: int
    num_inference_steps: int
    guidance_scale: float
    true_cfg_scale: float


def request_username(request: Optional[gr.Request]) -> Optional[str]:
    return getattr(request, "username", None) if request is not None else None


def current_user(request: Optional[gr.Request]) -> dict[str, Any]:
    try:
        return auth.require_user(request_username(request))
    except PermissionError as exc:
        raise gr.Error(str(exc)) from exc


def current_admin(request: Optional[gr.Request]) -> dict[str, Any]:
    try:
        return auth.require_admin(request_username(request))
    except PermissionError as exc:
        raise gr.Error(str(exc)) from exc


def user_jobs_root(user: dict[str, Any]) -> Path:
    root = USERS_OUTPUT_DIR / user["id"] / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def user_local_batch_root(user: dict[str, Any]) -> Path:
    root = LOCAL_BATCH_INPUT_DIR / "users" / user["id"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def now_utc() -> datetime:
    return datetime.now(timezone.utc)



def isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")



def session_paths(session_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        session_dir / METADATA_FILENAME,
        session_dir / IN_PROGRESS_MARKER,
        session_dir / COMPLETED_MARKER,
        session_dir / FAILED_MARKER,
    )



def create_session_dir(root_dir: Path, prefix: str) -> Path:
    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    session_dir = root_dir / f"{prefix}_{timestamp}_{suffix}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir



def write_session_metadata(session_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path, _, _, _ = session_paths(session_dir)
    metadata = dict(metadata)
    metadata["updated_at"] = isoformat_utc(now_utc())
    temp_path = metadata_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    temp_path.replace(metadata_path)



def read_session_metadata(session_dir: Path) -> dict[str, Any]:
    metadata_path, _, _, _ = session_paths(session_dir)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)



def batch_session_id(session_dir: Path) -> str:
    return session_dir.name



def batch_session_dir_from_id(session_id: str) -> Path:
    session_id = (session_id or "").strip()
    if not session_id:
        raise gr.Error("批量任务会话 ID 不能为空。")
    if "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
        raise gr.Error("无效的批量任务会话 ID。")
    session_dir = (BATCH_OUTPUT_DIR / session_id).resolve()
    ensure_relative_to_root(session_dir, BATCH_OUTPUT_DIR.resolve())
    return session_dir



def initialize_session(session_dir: Path, session_type: str, retention_days: int = RETENTION_DAYS) -> dict[str, Any]:
    created_at = now_utc()
    expire_at = created_at + timedelta(days=retention_days)
    metadata = {
        "session_type": session_type,
        "status": "in_progress" if session_type == "single" else BATCH_STATUS_PENDING,
        "created_at": isoformat_utc(created_at),
        "updated_at": isoformat_utc(created_at),
        "started_at": None,
        "finished_at": None,
        "expire_at": isoformat_utc(expire_at),
        "input_files": [],
        "result_files": [],
        "manifest_file": None,
        "uploaded_package_file": None,
        "extracted_package_dir": None,
        "batch_mode": None,
        "total_items": 0,
        "completed_items": 0,
        "success_items": 0,
        "failed_items": 0,
        "current_index": 0,
        "current_row_id": None,
        "current_phase": None,
        "progress": 0.0,
        "results_json_file": None,
        "results_csv_file": None,
        "results_zip_file": None,
        "download_ready": False,
        "last_error": None,
    }
    metadata_path, in_progress_path, completed_path, failed_path = session_paths(session_dir)
    completed_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    in_progress_path.touch()
    write_session_metadata(session_dir, metadata)
    return metadata



def finalize_session(session_dir: Path, metadata: dict[str, Any], status: str) -> None:
    metadata["status"] = status
    metadata["finished_at"] = isoformat_utc(now_utc())
    metadata["current_phase"] = "done" if status == BATCH_STATUS_COMPLETED else metadata.get("current_phase")
    write_session_metadata(session_dir, metadata)
    _, in_progress_path, completed_path, failed_path = session_paths(session_dir)
    in_progress_path.unlink(missing_ok=True)
    if status in {"completed", BATCH_STATUS_COMPLETED}:
        completed_path.touch()
        failed_path.unlink(missing_ok=True)
    else:
        failed_path.touch()
        completed_path.unlink(missing_ok=True)



def copy_file_to_dir(source: Path, target_dir: Path, target_name: Optional[str] = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / (target_name or source.name)
    shutil.copy2(source, destination)
    return destination



def local_path_from_input(file_obj) -> Path:
    if isinstance(file_obj, Path):
        return file_obj.resolve()
    return Path(getattr(file_obj, "name", file_obj)).resolve()



def write_partial_batch_results(session_dir: Path, rows: list[dict[str, Any]]) -> Path:
    partial_path = session_dir / PARTIAL_RESULTS_FILENAME
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return partial_path



def load_partial_batch_results(session_dir: Path) -> list[dict[str, Any]]:
    partial_path = session_dir / PARTIAL_RESULTS_FILENAME
    if not partial_path.exists():
        return []
    with open(partial_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []



def format_batch_progress_markdown(metadata: dict[str, Any]) -> str:
    status = metadata.get("status") or BATCH_STATUS_PENDING
    total_items = int(metadata.get("total_items") or 0)
    completed_items = int(metadata.get("completed_items") or 0)
    success_items = int(metadata.get("success_items") or 0)
    failed_items = int(metadata.get("failed_items") or 0)
    current_index = int(metadata.get("current_index") or 0)
    current_row_id = metadata.get("current_row_id") or "-"
    progress_value = float(metadata.get("progress") or 0.0)
    progress_percent = max(0.0, min(progress_value, 1.0)) * 100
    phase = metadata.get("current_phase") or "等待中"

    status_label_map = {
        BATCH_STATUS_PENDING: "等待启动",
        BATCH_STATUS_RUNNING: "正在处理",
        BATCH_STATUS_FINALIZING: "正在整理结果",
        BATCH_STATUS_COMPLETED: "已完成",
        BATCH_STATUS_FAILED: "已失败",
    }
    status_label = status_label_map.get(status, str(status))
    current_line = f"- 当前任务: {current_index}/{total_items}" if total_items else "- 当前任务: 0/0"
    if status in {BATCH_STATUS_COMPLETED, BATCH_STATUS_FAILED} and total_items:
        current_line = f"- 已处理: {completed_items}/{total_items}"

    return (
        "### 批量任务状态\n"
        f"- 状态: {status_label}\n"
        f"- 阶段: {phase}\n"
        f"- 进度: {progress_percent:.1f}%\n"
        f"{current_line}\n"
        f"- 当前条目 ID: {current_row_id}\n"
        f"- 成功: {success_items}\n"
        f"- 失败: {failed_items}"
    )



def format_batch_summary(metadata: dict[str, Any], session_dir: Path) -> str:
    total_items = int(metadata.get("total_items") or 0)
    success_items = int(metadata.get("success_items") or 0)
    failed_items = int(metadata.get("failed_items") or 0)
    results_zip = metadata.get("results_zip_file")
    summary = (
        "批量推理完成\n\n"
        f"- 总任务数: {total_items}\n"
        f"- 成功: {success_items}\n"
        f"- 失败: {failed_items}\n"
        f"- 结果目录: `{session_dir}`"
    )
    if results_zip:
        summary += f"\n- 下载文件: `{results_zip}`"
    return summary



def batch_status_payload(session_dir: Path, metadata: dict[str, Any]) -> tuple[str, str, Optional[str], bool, bool]:
    status = metadata.get("status")
    progress_markdown = format_batch_progress_markdown(metadata)
    summary = ""
    zip_path = None
    keep_polling = status in {BATCH_STATUS_PENDING, BATCH_STATUS_RUNNING, BATCH_STATUS_FINALIZING}
    allow_submit = not keep_polling
    if status == BATCH_STATUS_COMPLETED:
        summary = format_batch_summary(metadata, session_dir)
        zip_path = metadata.get("results_zip_file")
    elif status == BATCH_STATUS_FAILED:
        error_text = metadata.get("last_error") or "批量任务执行失败。"
        summary = f"批量推理失败\n\n- 任务目录: `{session_dir}`\n- 错误: {error_text}"
    return progress_markdown, summary, zip_path, keep_polling, allow_submit



def parse_expire_at(metadata: dict[str, Any]) -> Optional[datetime]:
    expire_at_value = metadata.get("expire_at")
    if not expire_at_value:
        return None
    try:
        return datetime.fromisoformat(str(expire_at_value).replace("Z", "+00:00"))
    except Exception:
        return None



def is_batch_session_recoverable(session_dir: Path, metadata: dict[str, Any]) -> bool:
    if metadata.get("session_type") not in {"batch_remote", "batch_local"}:
        return False
    expire_at = parse_expire_at(metadata)
    if expire_at is None or expire_at <= now_utc():
        return False
    status = metadata.get("status")
    if status == BATCH_STATUS_COMPLETED:
        zip_path = metadata.get("results_zip_file")
        if not zip_path or not Path(str(zip_path)).exists() or not metadata.get("download_ready"):
            return False
    return status in {
        BATCH_STATUS_PENDING,
        BATCH_STATUS_RUNNING,
        BATCH_STATUS_FINALIZING,
        BATCH_STATUS_COMPLETED,
        BATCH_STATUS_FAILED,
    }



def empty_batch_status_view(batch_mode: str = BATCH_MODE_REMOTE) -> tuple[str, str, Any, str, Any, Any, Any, Any, Any]:
    return (
        "",
        "",
        gr.update(value=None, visible=False),
        "",
        gr.update(active=False),
        gr.update(interactive=True),
        gr.update(value=batch_mode),
        batch_mode_help_text(batch_mode),
        gr.update(visible=batch_mode == BATCH_MODE_REMOTE, value=None),
    )



def cleanup_old_outputs() -> None:
    now = now_utc()
    LOGGER.info("Starting cleanup pass for retained outputs")
    deleted_sessions = 0
    skipped_in_progress = 0
    deleted_legacy_files = 0

    for session_root in (SINGLE_OUTPUT_DIR, BATCH_OUTPUT_DIR):
        for path in session_root.iterdir():
            if not path.is_dir():
                continue

            metadata_path, in_progress_path, completed_path, failed_path = session_paths(path)
            if in_progress_path.exists():
                skipped_in_progress += 1
                LOGGER.info("Skipping in-progress session: %s", path)
                continue
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                LOGGER.warning("Failed to read session metadata %s: %s", metadata_path, exc)
                continue

            session_type = metadata.get("session_type")
            if session_type not in {"single", "batch_remote"}:
                continue
            if not completed_path.exists() and not failed_path.exists():
                continue

            expire_at_value = metadata.get("expire_at")
            if not expire_at_value:
                continue
            expire_at = datetime.fromisoformat(str(expire_at_value).replace("Z", "+00:00"))
            if expire_at <= now:
                shutil.rmtree(path, ignore_errors=False)
                deleted_sessions += 1
                LOGGER.info("Deleted expired session: %s", path)

    for legacy_file in SINGLE_OUTPUT_DIR.iterdir():
        if legacy_file.is_dir():
            continue
        try:
            modified_at = datetime.fromtimestamp(legacy_file.stat().st_mtime, tz=timezone.utc)
        except FileNotFoundError:
            continue
        if modified_at + timedelta(days=RETENTION_DAYS) <= now:
            legacy_file.unlink(missing_ok=True)
            deleted_legacy_files += 1
            LOGGER.info("Deleted expired legacy single output: %s", legacy_file)

    LOGGER.info(
        "Cleanup pass completed: deleted_sessions=%s skipped_in_progress=%s deleted_legacy_files=%s",
        deleted_sessions,
        skipped_in_progress,
        deleted_legacy_files,
    )



def cleanup_loop() -> None:
    while True:
        try:
            cleanup_jobs.reconcile_interrupted_jobs()
            cleanup_jobs.cleanup_expired_jobs()
            cleanup_old_outputs()
        except Exception as exc:
            LOGGER.exception("Cleanup pass failed: %s", exc)
        time.sleep(CLEANUP_INTERVAL_SECONDS)



def register_active_batch_thread(session_id: str, worker: threading.Thread) -> None:
    with ACTIVE_BATCH_THREADS_LOCK:
        ACTIVE_BATCH_THREADS[session_id] = worker



def unregister_active_batch_thread(session_id: str) -> None:
    with ACTIVE_BATCH_THREADS_LOCK:
        ACTIVE_BATCH_THREADS.pop(session_id, None)



def is_active_batch_thread(session_id: str) -> bool:
    with ACTIVE_BATCH_THREADS_LOCK:
        worker = ACTIVE_BATCH_THREADS.get(session_id)
    return worker is not None and worker.is_alive()



def reconcile_in_progress_batch_sessions() -> None:
    for session_dir in BATCH_OUTPUT_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        metadata_path, in_progress_path, _, _ = session_paths(session_dir)
        if not metadata_path.exists() or not in_progress_path.exists():
            continue
        try:
            metadata = read_session_metadata(session_dir)
        except Exception as exc:
            LOGGER.warning("Failed to read batch session metadata during reconcile %s: %s", session_dir, exc)
            continue
        if metadata.get("session_type") not in {"batch_remote", "batch_local"}:
            continue
        if is_active_batch_thread(batch_session_id(session_dir)):
            continue
        metadata["status"] = BATCH_STATUS_FAILED
        metadata["current_phase"] = "任务中断"
        metadata["last_error"] = metadata.get("last_error") or "服务重启或任务线程中断，批量任务未完成。"
        finalize_session(session_dir, metadata, status=BATCH_STATUS_FAILED)
        LOGGER.info("Marked orphaned batch session as failed: %s", session_dir)



def start_cleanup_scheduler() -> None:
    cleanup_jobs.reconcile_interrupted_jobs()
    cleanup_jobs.cleanup_expired_jobs()
    reconcile_in_progress_batch_sessions()
    cleanup_old_outputs()
    cleanup_thread = threading.Thread(target=cleanup_loop, name="output-cleanup", daemon=True)
    cleanup_thread.start()



def detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def get_max_memory(include_cpu: bool = False) -> Optional[dict[Any, str]]:
    if not torch.cuda.is_available():
        return None

    max_memory: dict[Any, str] = {}
    gpu_count = torch.cuda.device_count()
    reserve_gib = float(os.getenv("GPU_MEMORY_RESERVE_GB", "2"))
    for gpu_idx in range(gpu_count):
        total_bytes = torch.cuda.get_device_properties(gpu_idx).total_memory
        total_gib = total_bytes / (1024 ** 3)
        usable_gib = max(8, int(total_gib - reserve_gib))
        max_memory[gpu_idx] = f"{usable_gib}GiB"
    if include_cpu:
        max_memory["cpu"] = os.getenv("CPU_OFFLOAD_MAX_MEMORY", "64GiB")
    return max_memory


def get_manual_device_map(gpu_count: int) -> Optional[dict[str, Any]]:
    if gpu_count < 4:
        return None

    return {
        "processor": 3,
        "scheduler": 3,
        "vae": 3,
        "text_encoder": 1,
        "transformer_head": 0,
        "transformer_tail": 2,
        "transformer_split": 30,
    }


def dispatch_transformer_layers(transformer, head_gpu: int, tail_gpu: int, split_index: int) -> None:
    transformer.pos_embed.to(f"cuda:{head_gpu}")
    transformer.time_text_embed.to(f"cuda:{head_gpu}")
    transformer.txt_norm.to(f"cuda:{head_gpu}")
    transformer.img_in.to(f"cuda:{head_gpu}")
    transformer.txt_in.to(f"cuda:{head_gpu}")

    for idx, block in enumerate(transformer.transformer_blocks):
        target_gpu = head_gpu if idx < split_index else tail_gpu
        block.to(f"cuda:{target_gpu}")

    transformer.norm_out.to(f"cuda:{tail_gpu}")
    transformer.proj_out.to(f"cuda:{tail_gpu}")


def load_pipeline_with_manual_dispatch(dtype: torch.dtype, force_cpu_offload: bool) -> QwenImageEditPlusPipeline:
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        local_files_only=True,
    )
    manual_device_map = get_manual_device_map(torch.cuda.device_count())
    if manual_device_map is None:
        pipeline.to("cuda")
        return pipeline

    dispatch_transformer_layers(
        pipeline.transformer,
        head_gpu=manual_device_map["transformer_head"],
        tail_gpu=manual_device_map["transformer_tail"],
        split_index=manual_device_map["transformer_split"],
    )
    pipeline.text_encoder.to(f"cuda:{manual_device_map['text_encoder']}")
    pipeline.vae.to(f"cuda:{manual_device_map['vae']}")
    if force_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    pipeline._manual_device_map = manual_device_map
    return pipeline


def describe_device_map(pipeline: QwenImageEditPlusPipeline) -> str:
    manual_device_map = getattr(pipeline, "_manual_device_map", None)
    if manual_device_map:
        parts = [
            f"text_encoder: cuda:{manual_device_map['text_encoder']}",
            f"vae: cuda:{manual_device_map['vae']}",
            f"transformer[0:{manual_device_map['transformer_split']}]: cuda:{manual_device_map['transformer_head']}",
            f"transformer[{manual_device_map['transformer_split']}:60]: cuda:{manual_device_map['transformer_tail']}",
        ]
        return " | ".join(parts)
    hf_device_map = getattr(pipeline, "hf_device_map", None)
    if not hf_device_map:
        return _DEVICE or "unknown"
    summary: dict[str, list[str]] = {}
    for module_name, device in hf_device_map.items():
        summary.setdefault(str(device), []).append(module_name)
    parts = []
    for device, modules in summary.items():
        parts.append(f"{device}: {', '.join(modules[:4])}{' ...' if len(modules) > 4 else ''}")
    return " | ".join(parts)


def enable_memory_optimizations(pipeline: QwenImageEditPlusPipeline, use_memory_saving: bool) -> None:
    if use_memory_saving and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()


def patch_manual_pipeline_devices(pipeline: QwenImageEditPlusPipeline) -> QwenImageEditPlusPipeline:
    manual_device_map = getattr(pipeline, "_manual_device_map", None)
    if not manual_device_map:
        return pipeline

    vae_device = torch.device(f"cuda:{manual_device_map['vae']}")
    text_device = torch.device(f"cuda:{manual_device_map['text_encoder']}")
    transformer_head_device = torch.device(f"cuda:{manual_device_map['transformer_head']}")
    transformer_tail_device = torch.device(f"cuda:{manual_device_map['transformer_tail']}")
    split_index = manual_device_map["transformer_split"]

    original_prepare_latents = pipeline.prepare_latents
    original_encode_prompt = pipeline.encode_prompt
    original_scheduler_step = pipeline.scheduler.step
    original_set_timesteps = pipeline.scheduler.set_timesteps
    original_vae_decode = pipeline.vae.decode
    original_call = pipeline.__class__.__call__

    def prepare_latents_on_vae(images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        latents_out, image_latents = original_prepare_latents(
            images,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            vae_device,
            generator,
            latents,
        )
        target_dtype = pipeline.transformer.img_in.weight.dtype
        latents_out = latents_out.to(device=transformer_head_device, dtype=target_dtype)
        if image_latents is not None:
            image_latents = image_latents.to(device=transformer_head_device, dtype=target_dtype)
        return latents_out, image_latents

    def encode_prompt_on_text(prompt, image=None, device=None, num_images_per_prompt=1, prompt_embeds=None, prompt_embeds_mask=None, max_sequence_length=1024):
        return original_encode_prompt(
            prompt=prompt,
            image=image,
            device=text_device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )

    def transformer_forward_on_shards(
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict=True,
    ):
        transformer = pipeline.transformer

        transformer_dtype = transformer.img_in.weight.dtype
        hidden_states = hidden_states.to(device=transformer_head_device, dtype=transformer_dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device=transformer_head_device, dtype=transformer_dtype)
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(transformer_head_device)
        if timestep is not None:
            timestep = timestep.to(device=transformer_head_device, dtype=transformer_dtype)
        if guidance is not None:
            guidance = guidance.to(device=transformer_head_device, dtype=torch.float32)
        if additional_t_cond is not None and hasattr(additional_t_cond, "to"):
            additional_t_cond = additional_t_cond.to(device=transformer_head_device, dtype=transformer_dtype)

        hidden_states = transformer.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if transformer.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * int(torch.tensor(sample[0]).prod().item()) + [1] * sum(int(torch.tensor(s).prod().item()) for s in sample[1:]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = transformer.txt_norm(encoder_hidden_states)
        encoder_hidden_states = transformer.txt_in(encoder_hidden_states)
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            transformer.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else transformer.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )
        image_rotary_emb = transformer.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            joint_attention_mask = joint_attention_mask[:, None, None, :]
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        for index_block, block in enumerate(transformer.transformer_blocks):
            target_device = transformer_head_device if index_block < split_index else transformer_tail_device
            if hidden_states.device != target_device:
                hidden_states = hidden_states.to(device=target_device, dtype=transformer_dtype)
                encoder_hidden_states = encoder_hidden_states.to(device=target_device, dtype=transformer_dtype)
                temb = temb.to(device=target_device, dtype=transformer_dtype)
                if image_rotary_emb is not None:
                    image_rotary_emb = tuple(item.to(target_device) for item in image_rotary_emb)
                if block_attention_kwargs.get("attention_mask") is not None:
                    block_attention_kwargs["attention_mask"] = block_attention_kwargs["attention_mask"].to(target_device)
                if modulate_index is not None:
                    modulate_index = modulate_index.to(target_device)
                print(f"[manual] switch transformer shard at block={index_block} -> {target_device}", flush=True)

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

            if controlnet_block_samples is not None:
                interval_control = len(transformer.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(torch.ceil(torch.tensor(interval_control)).item())
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control].to(hidden_states.device)

        if transformer.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = transformer.norm_out(hidden_states, temb)
        output = transformer.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @functools.wraps(original_set_timesteps)
    def set_timesteps_on_tail(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = transformer_tail_device
        elif len(args) >= 2:
            args = list(args)
            args[1] = transformer_tail_device
        result = original_set_timesteps(*args, **kwargs)
        if hasattr(pipeline.scheduler, "timesteps"):
            pipeline.scheduler.timesteps = pipeline.scheduler.timesteps.to(transformer_tail_device)
        if hasattr(pipeline.scheduler, "sigmas"):
            pipeline.scheduler.sigmas = pipeline.scheduler.sigmas.to(transformer_tail_device)
        return result

    def scheduler_step_on_tail(model_output, timestep, sample, *args, **kwargs):
        model_output = model_output.to(transformer_tail_device, dtype=torch.float32)
        sample = sample.to(transformer_tail_device, dtype=torch.float32)
        if hasattr(timestep, "to"):
            timestep = timestep.to(transformer_tail_device)
        if hasattr(pipeline.scheduler, "timesteps"):
            pipeline.scheduler.timesteps = pipeline.scheduler.timesteps.to(transformer_tail_device)
        if hasattr(pipeline.scheduler, "sigmas"):
            pipeline.scheduler.sigmas = pipeline.scheduler.sigmas.to(transformer_tail_device)
        if "per_token_timesteps" in kwargs and kwargs["per_token_timesteps"] is not None:
            kwargs["per_token_timesteps"] = kwargs["per_token_timesteps"].to(transformer_tail_device)
        result = original_scheduler_step(model_output, timestep, sample, *args, **kwargs)
        if isinstance(result, tuple) and result:
            return (result[0].to(transformer_tail_device), *result[1:])
        if hasattr(result, "prev_sample"):
            result.prev_sample = result.prev_sample.to(transformer_tail_device)
        return result

    def vae_decode_on_vae(latents, *args, **kwargs):
        return original_vae_decode(latents.to(vae_device), *args, **kwargs)

    @functools.wraps(original_call)
    def manual_call(
        self,
        image=None,
        prompt=None,
        negative_prompt=None,
        true_cfg_scale=4.0,
        height=None,
        width=None,
        num_inference_steps=50,
        sigmas=None,
        guidance_scale=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        output_type="pil",
        return_dict=True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=512,
    ):
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if batch_size > 1:
            raise ValueError(
                f"QwenImageEditPlusPipeline currently only supports batch_size=1, but received batch_size={batch_size}. "
                "Please process prompts one at a time."
            )

        device = transformer_head_device
        condition_images = None
        vae_image_sizes = []
        vae_images = None
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_images = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        sigmas = torch.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist() if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            transformer_tail_device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=transformer_head_device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    image_latents = image_latents.to(device=latents.device, dtype=latents.dtype)
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.device, latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                if i == 0:
                    print(
                        f"[manual] step={i} latent={tuple(latents.shape)} {latents.device} {latents.dtype} | "
                        f"image_latents={None if image_latents is None else (tuple(image_latents.shape), image_latents.device, image_latents.dtype)} | "
                        f"prompt={prompt_embeds.device} {prompt_embeds.dtype}",
                        flush=True,
                    )
                elif i % 5 == 0:
                    print(f"[manual] step={i}/{len(timesteps)} t={float(t):.6f} latents={latents.device} {latents.dtype}", flush=True)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents.to(transformer_tail_device), height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)

    pipeline.prepare_latents = prepare_latents_on_vae
    pipeline.encode_prompt = encode_prompt_on_text
    pipeline.transformer.forward = transformer_forward_on_shards
    pipeline.scheduler.set_timesteps = set_timesteps_on_tail
    pipeline.scheduler.step = scheduler_step_on_tail
    pipeline.vae.decode = vae_decode_on_vae
    pipeline.__class__.__call__ = manual_call
    return pipeline



def get_pipeline_execution_device(pipeline: QwenImageEditPlusPipeline) -> str:
    manual_device_map = getattr(pipeline, "_manual_device_map", None)
    if manual_device_map:
        return f"cuda:{manual_device_map['vae']}"
    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is not None:
        return str(execution_device)
    return _DEVICE or "cpu"


def get_pipeline() -> QwenImageEditPlusPipeline:
    global _PIPELINE, _DEVICE, _DTYPE, _DEVICE_MAP_INFO, _PIPELINE_EXECUTION_DEVICE
    if _PIPELINE is None:
        device, dtype = detect_device()
        _DEVICE = device
        _DTYPE = dtype
        if device == "cuda":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            gpu_count = torch.cuda.device_count()
            force_cpu_offload = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
            use_memory_saving = os.getenv("ENABLE_ATTENTION_SLICING", "0") == "1"
            device_map_mode = os.getenv("DEVICE_MAP", "manual")
            if gpu_count > 1 and device_map_mode == "manual":
                _PIPELINE = load_pipeline_with_manual_dispatch(dtype=dtype, force_cpu_offload=force_cpu_offload)
            else:
                from_pretrained_kwargs = {
                    "torch_dtype": dtype,
                    "local_files_only": True,
                }
                if gpu_count > 1:
                    from_pretrained_kwargs["device_map"] = device_map_mode
                    from_pretrained_kwargs["max_memory"] = get_max_memory(include_cpu=force_cpu_offload)
                _PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
                    str(MODEL_DIR),
                    **from_pretrained_kwargs,
                )
                if gpu_count <= 1:
                    _PIPELINE.to(device)
                    if force_cpu_offload and hasattr(_PIPELINE, "enable_model_cpu_offload"):
                        _PIPELINE.enable_model_cpu_offload()
            enable_memory_optimizations(_PIPELINE, use_memory_saving=use_memory_saving)
            _PIPELINE = patch_manual_pipeline_devices(_PIPELINE)
            _DEVICE_MAP_INFO = describe_device_map(_PIPELINE)
            _PIPELINE_EXECUTION_DEVICE = get_pipeline_execution_device(_PIPELINE)
        else:
            _PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
                str(MODEL_DIR),
                torch_dtype=dtype,
                local_files_only=True,
            )
            _PIPELINE.to(device)
            _DEVICE_MAP_INFO = device
            _PIPELINE_EXECUTION_DEVICE = device
        _PIPELINE.set_progress_bar_config(disable=True)
    return _PIPELINE


def parse_uploaded_images(files) -> List[Path]:
    if not files:
        return []
    parsed = []
    for file in files:
        file_path = local_path_from_input(file)
        if file_path.exists():
            parsed.append(file_path)
    return parsed


def load_images(paths: List[Path]) -> List[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def sanitize_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned[:80] or "result"


def get_primary_input_stem(image_paths: List[Path]) -> str:
    if not image_paths:
        return "image"
    return sanitize_filename(image_paths[0].stem)


def run_generation(
    image_paths: List[Path],
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Image.Image:
    if not image_paths:
        raise gr.Error("请至少上传一张输入图片。")
    if not prompt.strip():
        raise gr.Error("请输入 prompt。")

    width = int(width) if width not in (None, 0, "") else None
    height = int(height) if height not in (None, 0, "") else None
    if (width is None) != (height is None):
        raise gr.Error("宽度和高度需要同时填写，或同时留空使用自动分辨率。")
    if width is not None and (width <= 0 or height <= 0):
        raise gr.Error("宽度和高度必须为正整数。")

    pipeline = get_pipeline()
    generator_device = _PIPELINE_EXECUTION_DEVICE or _DEVICE or "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    images = load_images(image_paths)
    with torch.inference_mode():
        result = pipeline(
            image=images,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            width=width,
            height=height,
        )
    return result.images[0]


def save_single_result(image: Image.Image, session_dir: Path, image_paths: List[Path], seed: int) -> str:
    input_stem = get_primary_input_stem(image_paths)
    file_name = f"{input_stem}_seed{seed}.png"
    output_path = session_dir / file_name
    image.save(output_path)
    return str(output_path)



def persist_single_inputs(image_paths: List[Path], session_dir: Path) -> List[Path]:
    inputs_dir = session_dir / "inputs"
    persisted_paths = []
    for index, source_path in enumerate(image_paths, start=1):
        persisted_name = f"{index:03d}_{source_path.name}"
        persisted_paths.append(copy_file_to_dir(source_path, inputs_dir, persisted_name))
    return persisted_paths



def infer_single(
    images,
    prompt,
    negative_prompt,
    seed,
    num_inference_steps,
    guidance_scale,
    true_cfg_scale,
    width,
    height,
    request: gr.Request,
):
    user = current_user(request)
    image_paths = parse_uploaded_images(images)
    session_dir = create_session_dir(user_jobs_root(user), "qwen_image_edit_single")
    job = task_store.create_job(user["id"], "single", session_dir, status=task_store.JOB_STATUS_RUNNING)
    metadata = initialize_session(session_dir, session_type="single")
    metadata["user_id"] = user["id"]
    metadata["username"] = user["username"]
    metadata["job_id"] = job["id"]
    try:
        persisted_inputs = persist_single_inputs(image_paths, session_dir)
        metadata["input_files"] = [str(path) for path in persisted_inputs]
        task_store.update_job(job["id"], current_phase="正在推理", heartbeat_at=isoformat_utc(now_utc()))
        result = run_generation(
            image_paths=image_paths,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
        )
        saved_path = save_single_result(result, session_dir, image_paths, seed)
        metadata["result_files"] = [saved_path]
        task_store.finish_job(
            job["id"],
            task_store.JOB_STATUS_COMPLETED,
            single_result_file=saved_path,
            download_ready=1,
            progress=1.0,
            completed_items=1,
            success_items=1,
            current_phase="处理完成",
        )
        finalize_session(session_dir, metadata, status="completed")
    except Exception as exc:
        metadata["last_error"] = str(exc)
        task_store.finish_job(
            job["id"],
            task_store.JOB_STATUS_FAILED,
            last_error=str(exc),
            current_phase="任务失败",
        )
        finalize_session(session_dir, metadata, status="failed")
        raise

    resolution_text = "自动" if not width and not height else f"{int(width)} × {int(height)}"
    status = (
        f"推理完成\n\n"
        f"- 任务 ID: `{job['id']}`\n"
        f"- 用户: `{user['username']}`\n"
        f"- device: `{_DEVICE}`\n"
        f"- dtype: `{_DTYPE}`\n"
        f"- device_map: `{_DEVICE_MAP_INFO}`\n"
        f"- attention_slicing: `{os.getenv('ENABLE_ATTENTION_SLICING', '0')}`\n"
        f"- cpu_offload: `{os.getenv('ENABLE_CPU_OFFLOAD', '0')}`\n"
        f"- 输入图片数: {len(image_paths)}\n"
        f"- 输出分辨率: `{resolution_text}`\n"
        f"- 输出目录: `{session_dir}`\n"
        f"- 输出文件: `{saved_path}`"
    )
    return result, status, saved_path


def load_batch_manifest_rows(file_obj) -> tuple[Path, List[dict[str, Any]]]:
    if file_obj is None:
        raise gr.Error("请上传批量任务文件（CSV 或 JSON）。")

    file_path = local_path_from_input(file_obj)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
            if not isinstance(rows, list):
                raise gr.Error("JSON 须为数组，每个元素代表一条任务。")
    else:
        raise gr.Error("仅支持 CSV 或 JSON 格式的批量任务文件。")

    normalized_rows = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise gr.Error(f"批量任务第 {index} 条不是对象/字典。")
        normalized_rows.append(row)
    return file_path, normalized_rows


def parse_row_image_refs(row: dict[str, Any]) -> List[str]:
    raw_images = row.get("images") or row.get("image") or ""
    if isinstance(raw_images, list):
        return [str(item).strip() for item in raw_images if str(item).strip()]
    return [part.strip() for part in str(raw_images).split("|") if part.strip()]


def ensure_relative_to_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise gr.Error(f"上传图片包中的路径越界: {path}") from exc



def persist_batch_uploads(session_dir: Path, manifest_file, image_package=None) -> tuple[Path, Optional[Path]]:
    uploads_dir = session_dir / "uploads"
    manifest_path = local_path_from_input(manifest_file)
    persisted_manifest = copy_file_to_dir(manifest_path, uploads_dir)
    persisted_package = None
    if image_package is not None:
        package_path = local_path_from_input(image_package)
        persisted_package = copy_file_to_dir(package_path, uploads_dir)
    return persisted_manifest, persisted_package



def looks_like_absolute_image_ref(image_ref: str) -> bool:
    path = Path(image_ref)
    windows_path = PureWindowsPath(image_ref)
    return path.is_absolute() or windows_path.is_absolute() or bool(windows_path.drive)



def normalize_relative_image_ref(image_ref: str, mode_label: str) -> PurePosixPath:
    normalized_ref = PurePosixPath(image_ref)
    if normalized_ref.is_absolute() or ".." in normalized_ref.parts:
        raise gr.Error(f"{mode_label}不允许越界路径: {image_ref}")
    return normalized_ref



def resolve_batch_image_path(
    image_ref: str,
    batch_mode: str,
    manifest_dir: Path,
    package_root: Optional[Path] = None,
    local_batch_root: Path = LOCAL_BATCH_INPUT_DIR,
) -> Path:
    image_ref = image_ref.strip()
    if not image_ref:
        raise gr.Error("批量任务中的 images 字段不能为空。")

    if looks_like_absolute_image_ref(image_ref):
        if batch_mode == BATCH_MODE_REMOTE:
            raise gr.Error(
                "远程上传模式下，任务文件中的 images 不能使用绝对路径。"
                "请使用 generate_batch_manifest.py 的 --image-path-mode package-relative 重新生成任务文件。"
            )
        raise gr.Error(
            "服务端本地图片模式下，任务文件中的 images 不能使用绝对路径。"
            f"请先把图片放到项目目录 `{LOCAL_BATCH_INPUT_DIR.name}/` 下，再使用文件名或相对路径。"
        )

    if batch_mode == BATCH_MODE_REMOTE:
        if package_root is None:
            raise gr.Error("远程上传模式必须上传图片包 ZIP。")
        normalized_ref = normalize_relative_image_ref(image_ref, "远程上传模式")
        resolved_path = (package_root / Path(*normalized_ref.parts)).resolve()
        ensure_relative_to_root(resolved_path, package_root)
        return resolved_path

    normalized_ref = normalize_relative_image_ref(image_ref, "服务端本地图片模式")
    resolved_path = (local_batch_root / Path(*normalized_ref.parts)).resolve()
    ensure_relative_to_root(resolved_path, local_batch_root.resolve())
    return resolved_path


def prepare_batch_image_package(package_file, session_dir: Path) -> Optional[Path]:
    if package_file is None:
        return None

    package_path = local_path_from_input(package_file)
    if package_path.suffix.lower() != ".zip":
        raise gr.Error("图片包仅支持 ZIP 格式。")

    package_root = session_dir / "input_package"
    package_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(package_path) as zf:
            if not zf.namelist():
                raise gr.Error("上传的图片包为空。")
            for member in zf.infolist():
                member_path = PurePosixPath(member.filename)
                if not member.filename or member.filename.endswith("/"):
                    continue
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise gr.Error(f"图片包中存在非法路径: {member.filename}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise gr.Error(f"图片包中不允许符号链接: {member.filename}")
                target_path = (package_root / Path(*member_path.parts)).resolve()
                ensure_relative_to_root(target_path, package_root)
            zf.extractall(package_root)
    except zipfile.BadZipFile as exc:
        raise gr.Error("上传的图片包不是有效的 ZIP 文件。") from exc

    return package_root


def parse_batch_manifest(
    file_obj,
    batch_mode: str,
    package_root: Optional[Path] = None,
    local_batch_root: Path = LOCAL_BATCH_INPUT_DIR,
) -> List[BatchItem]:
    manifest_path, rows = load_batch_manifest_rows(file_obj)
    manifest_dir = manifest_path.parent

    items: List[BatchItem] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        prompt = str(row.get("prompt") or "").strip()
        negative_prompt = str(row.get("negative_prompt") or " ")
        image_refs = parse_row_image_refs(row)
        image_paths = [
            resolve_batch_image_path(
                image_ref,
                batch_mode=batch_mode,
                manifest_dir=manifest_dir,
                package_root=package_root,
                local_batch_root=local_batch_root,
            )
            for image_ref in image_refs
        ]
        for image_ref, path in zip(image_refs, image_paths):
            if not path.exists():
                raise gr.Error(f"批量任务 {row_id} 的图片不存在: {image_ref}")
        items.append(
            BatchItem(
                row_id=row_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_refs=image_refs,
                image_paths=image_paths,
                seed=int(row.get("seed", 0) or 0),
                num_inference_steps=int(row.get("num_inference_steps", 40) or 40),
                guidance_scale=float(row.get("guidance_scale", 1.0) or 1.0),
                true_cfg_scale=float(row.get("true_cfg_scale", 4.0) or 4.0),
            )
        )
    return items


def batch_examples_markdown() -> str:
    csv_example = """id,prompt,negative_prompt,images,seed,num_inference_steps,guidance_scale,true_cfg_scale
1,A silver robot standing in a flower field., ,task_001/input1.png,0,40,1.0,4.0
2,Merge the two people into one travel photo., ,task_001/a.png|task_001/b.png,42,40,1.0,4.0"""
    json_example = [
        {
            "id": "1",
            "prompt": "A silver robot standing in a flower field.",
            "negative_prompt": " ",
            "images": ["task_001/input1.png"],
            "seed": 0,
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
    ]
    return (
        "### 批量任务文件格式\n"
        "支持两种模式：\n"
        f"- **{BATCH_MODE_LOCAL}**：先把图片放到项目目录 `{LOCAL_BATCH_INPUT_DIR.name}/` 下，`images` 只写文件名或相对该目录的路径。\n"
        f"- **{BATCH_MODE_REMOTE}**：上传 `CSV/JSON` + `ZIP` 图片包，`images` 必须写成相对 `ZIP` 根目录的路径。\n"
        "多图输入在 CSV 中使用 `|` 分隔。远程上传模式建议用 `generate_batch_manifest.py --image-path-mode package-relative` 生成任务文件；服务端本地图片模式建议用 `--image-path-mode project-relative`。\n"
        "批量推理界面仅展示总进度，并在全部任务完成后提供一个最终结果 ZIP 下载入口。\n"
        "单次推理与远程批量上传文件/结果会在服务器暂存 7 天，之后自动清理。\n\n"
        "**CSV 示例**\n"
        f"```csv\n{csv_example}\n```\n"
        "**JSON 示例**\n"
        f"```json\n{json.dumps(json_example, ensure_ascii=False, indent=2)}\n```"
    )



def batch_mode_help_text(batch_mode: str) -> str:
    if batch_mode == BATCH_MODE_REMOTE:
        return (
            "**当前模式：远程上传**\n\n"
            "- 上传 `manifest + ZIP`。\n"
            "- `images` 必须写成 ZIP 内相对路径。\n"
            "- 服务端会先把 manifest 和 ZIP 保存到本次会话目录，再从解压目录读取图片。"
        )
    return (
        "**当前模式：服务端本地图片**\n\n"
        f"- 请先把图片手动放到项目目录 `{LOCAL_BATCH_INPUT_DIR.name}/` 下。\n"
        "- `images` 只能写文件名或相对该目录的路径。\n"
        "- 不支持绝对路径，也不支持项目目录之外的图片路径。"
    )



def update_batch_mode_ui(batch_mode: str):
    is_remote = batch_mode == BATCH_MODE_REMOTE
    return batch_mode_help_text(batch_mode), gr.update(visible=is_remote, value=None)



def reset_batch_outputs(batch_mode: str = BATCH_MODE_REMOTE):
    return empty_batch_status_view(batch_mode=batch_mode)



def job_metadata(job: dict[str, Any]) -> dict[str, Any]:
    status = job.get("status")
    session_type = job.get("job_type")
    metadata = {
        "session_type": session_type,
        "user_id": job.get("user_id"),
        "job_id": job.get("id"),
        "status": status,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "expire_at": job.get("retention_expire_at"),
        "input_files": [],
        "result_files": [],
        "manifest_file": job.get("manifest_file"),
        "uploaded_package_file": job.get("uploaded_package_file"),
        "extracted_package_dir": job.get("extracted_package_dir"),
        "batch_mode": job.get("batch_mode"),
        "total_items": job.get("total_items") or 0,
        "completed_items": job.get("completed_items") or 0,
        "success_items": job.get("success_items") or 0,
        "failed_items": job.get("failed_items") or 0,
        "current_index": job.get("current_index") or 0,
        "current_row_id": job.get("current_row_id"),
        "current_phase": job.get("current_phase"),
        "progress": job.get("progress") or 0.0,
        "results_json_file": job.get("results_json_file"),
        "results_csv_file": job.get("results_csv_file"),
        "results_zip_file": job.get("results_zip_file"),
        "download_ready": bool(job.get("download_ready")),
        "last_error": job.get("last_error"),
    }
    if job.get("single_result_file"):
        metadata["result_files"] = [job["single_result_file"]]
    if job.get("results_zip_file"):
        metadata["result_files"] = [path for path in [job.get("results_csv_file"), job.get("results_json_file"), job.get("results_zip_file")] if path]
    return metadata


def write_job_metadata(job: dict[str, Any]) -> None:
    session_dir = Path(job["session_dir"])
    write_session_metadata(session_dir, job_metadata(job))


def execute_batch_session(job_id: str) -> None:
    result_files: list[str] = []
    try:
        job = task_store.get_job(job_id)
        if not job:
            raise RuntimeError(f"批量任务不存在: {job_id}")
        session_dir = Path(job["session_dir"])
        task_store.update_job(
            job_id,
            status=task_store.JOB_STATUS_RUNNING,
            started_at=job.get("started_at") or isoformat_utc(now_utc()),
            current_phase="正在处理",
            heartbeat_at=isoformat_utc(now_utc()),
        )
        write_job_metadata(task_store.get_job(job_id))

        items = task_store.list_job_items(job_id)
        total_items = len(items)
        for item in items:
            if item["status"] in {task_store.JOB_ITEM_SUCCESS, task_store.JOB_ITEM_FAILED}:
                continue
            idx = int(item["row_index"])
            image_paths = [Path(path) for path in task_store.loads_json(item["resolved_image_paths_json"], [])]
            image_refs = task_store.loads_json(item["image_refs_json"], [])
            task_store.update_job_item(item["id"], status=task_store.JOB_ITEM_RUNNING, started_at=isoformat_utc(now_utc()))
            task_store.update_job(
                job_id,
                current_index=idx,
                current_row_id=item["row_id"],
                current_phase=f"批量处理中 {idx}/{total_items}",
                heartbeat_at=isoformat_utc(now_utc()),
            )
            write_job_metadata(task_store.get_job(job_id))
            try:
                result = run_generation(
                    image_paths=image_paths,
                    prompt=item["prompt"],
                    negative_prompt=item["negative_prompt"],
                    seed=item["seed"],
                    num_inference_steps=item["num_inference_steps"],
                    guidance_scale=item["guidance_scale"],
                    true_cfg_scale=item["true_cfg_scale"],
                )
                input_stem = get_primary_input_stem(image_paths)
                output_path = session_dir / f"{idx:03d}_{input_stem}.png"
                result.save(output_path)
                result_files.append(str(output_path))
                task_store.update_job_item(
                    item["id"],
                    status=task_store.JOB_ITEM_SUCCESS,
                    output_image=str(output_path),
                    error="",
                    traceback="",
                    finished_at=isoformat_utc(now_utc()),
                )
            except Exception as exc:
                error_traceback = traceback.format_exc()
                LOGGER.error(
                    "Batch item failed: job=%s row_id=%s images=%s\n%s",
                    job_id,
                    item["row_id"],
                    "|".join(image_refs),
                    error_traceback,
                )
                task_store.update_job_item(
                    item["id"],
                    status=task_store.JOB_ITEM_FAILED,
                    output_image="",
                    error=str(exc),
                    traceback=error_traceback if os.getenv("INCLUDE_TRACEBACK_IN_RESULTS", "0") == "1" else "",
                    finished_at=isoformat_utc(now_utc()),
                )

            task_store.update_job_progress_from_items(job_id)
            rows = task_store.rows_for_batch_results(job_id)
            write_partial_batch_results(session_dir, rows)
            write_job_metadata(task_store.get_job(job_id))

        task_store.update_job(job_id, status=task_store.JOB_STATUS_FINALIZING, current_phase="正在整理结果", current_row_id=None)
        write_job_metadata(task_store.get_job(job_id))

        rows = task_store.rows_for_batch_results(job_id)
        csv_path = session_dir / "batch_results.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "prompt",
                    "negative_prompt",
                    "images",
                    "seed",
                    "num_inference_steps",
                    "guidance_scale",
                    "true_cfg_scale",
                    "status",
                    "output_image",
                    "error",
                    "traceback",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        json_path = session_dir / "batch_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        zip_path = session_dir / "batch_results.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
            zf.write(json_path, arcname=json_path.name)
            for row in rows:
                if row["output_image"]:
                    zf.write(row["output_image"], arcname=Path(row["output_image"]).name)

        task_store.finish_job(
            job_id,
            task_store.JOB_STATUS_COMPLETED,
            progress=1.0,
            current_index=total_items,
            current_phase="处理完成",
            results_csv_file=str(csv_path),
            results_json_file=str(json_path),
            results_zip_file=str(zip_path),
            download_ready=1,
        )
        write_job_metadata(task_store.get_job(job_id))
    except Exception as exc:
        task_store.finish_job(job_id, task_store.JOB_STATUS_FAILED, last_error=str(exc), current_phase="任务失败")
        job = task_store.get_job(job_id)
        if job:
            write_job_metadata(job)
        LOGGER.exception("Batch session failed: %s", job_id)
    finally:
        unregister_active_batch_thread(job_id)



def batch_status_payload_from_job(job: dict[str, Any]) -> tuple[str, str, Optional[str], bool, bool]:
    metadata = job_metadata(job)
    progress_markdown = format_batch_progress_markdown(metadata)
    status = job.get("status")
    keep_polling = status in {task_store.JOB_STATUS_QUEUED, task_store.JOB_STATUS_RUNNING, task_store.JOB_STATUS_FINALIZING}
    allow_submit = not keep_polling
    summary = ""
    zip_path = None
    if status == task_store.JOB_STATUS_COMPLETED:
        summary = format_batch_summary(metadata, Path(job["session_dir"]))
        candidate = job.get("results_zip_file")
        if job.get("download_ready") and candidate and Path(candidate).exists():
            zip_path = candidate
    elif status in {task_store.JOB_STATUS_FAILED, task_store.JOB_STATUS_INTERRUPTED}:
        error_text = job.get("last_error") or "批量任务执行失败。"
        summary = f"批量推理{('中断' if status == task_store.JOB_STATUS_INTERRUPTED else '失败')}\n\n- 任务目录: `{job['session_dir']}`\n- 错误: {error_text}"
    return progress_markdown, summary, zip_path, keep_polling, allow_submit


def load_batch_session_status(
    session_id: str,
    *,
    restore_on_load: bool,
    default_batch_mode: str = BATCH_MODE_REMOTE,
    request: gr.Request = None,
) -> tuple[str, str, Any, str, Any, Any, Any, Any, Any]:
    if not session_id:
        return empty_batch_status_view(batch_mode=default_batch_mode)

    user = current_user(request)
    job = task_store.get_job_for_user(session_id, user)
    if not job:
        return empty_batch_status_view(batch_mode=default_batch_mode)

    mode_value = job.get("batch_mode") or default_batch_mode
    progress_markdown, summary, zip_path, keep_polling, allow_submit = batch_status_payload_from_job(job)
    zip_update = gr.update(value=zip_path, visible=bool(zip_path)) if zip_path else gr.update(value=None, visible=False)
    return (
        progress_markdown,
        summary,
        zip_update,
        session_id,
        gr.update(active=keep_polling),
        gr.update(interactive=allow_submit),
        gr.update(value=mode_value),
        batch_mode_help_text(mode_value),
        gr.update(visible=mode_value == BATCH_MODE_REMOTE, value=None),
    )



def poll_batch_status(session_id: str, batch_mode: str, request: gr.Request):
    return load_batch_session_status(session_id, restore_on_load=False, default_batch_mode=batch_mode, request=request)



def restore_batch_session_on_load(session_id: str, batch_mode: str, request: gr.Request):
    return load_batch_session_status(session_id, restore_on_load=True, default_batch_mode=batch_mode, request=request)



def format_jobs_table(jobs: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for job in jobs:
        rows.append(
            [
                job.get("id"),
                job.get("username", ""),
                job.get("job_type"),
                job.get("status"),
                f"{float(job.get('progress') or 0.0) * 100:.1f}%",
                job.get("created_at"),
                job.get("finished_at") or "",
                "可下载" if job.get("status") == task_store.JOB_STATUS_COMPLETED and job.get("download_ready") else "不可下载",
            ]
        )
    return rows


def refresh_my_jobs(request: gr.Request) -> list[list[Any]]:
    user = current_user(request)
    return format_jobs_table(task_store.list_jobs_for_user(user))


def load_job_detail(job_id: str, request: gr.Request) -> tuple[str, Any]:
    user = current_user(request)
    job_id = (job_id or "").strip()
    if not job_id:
        return "请输入任务 ID。", gr.update(value=None, visible=False)
    job = task_store.get_job_for_user(job_id, user)
    if not job:
        return "任务不存在，或你没有权限查看。", gr.update(value=None, visible=False)

    metadata = job_metadata(job)
    detail = format_batch_progress_markdown(metadata) if job["job_type"].startswith("batch") else (
        "### 单次任务状态\n"
        f"- 状态: {job['status']}\n"
        f"- 进度: {float(job.get('progress') or 0.0) * 100:.1f}%\n"
        f"- 阶段: {job.get('current_phase') or '-'}\n"
        f"- 错误: {job.get('last_error') or '-'}"
    )
    detail += f"\n- 任务 ID: `{job['id']}`\n- 任务目录: `{job['session_dir']}`"
    download_path = job.get("results_zip_file") if job["job_type"].startswith("batch") else job.get("single_result_file")
    can_download = job.get("status") == task_store.JOB_STATUS_COMPLETED and bool(job.get("download_ready")) and download_path and Path(download_path).exists()
    return detail, gr.update(value=download_path if can_download else None, visible=bool(can_download))


def admin_refresh_users(request: gr.Request) -> list[list[Any]]:
    user = current_user(request)
    if user.get("role") != task_store.ROLE_ADMIN:
        return []
    return [
        [
            user["id"],
            user["username"],
            user["role"],
            "启用" if user["is_active"] else "禁用",
            user["created_at"],
            user.get("last_login_at") or "",
        ]
        for user in task_store.list_users()
    ]


def admin_create_user(username: str, password: str, role: str, request: gr.Request) -> tuple[str, list[list[Any]]]:
    admin = current_admin(request)
    role = role or task_store.ROLE_USER
    if role not in task_store.USER_ROLES:
        raise gr.Error("无效用户角色。")
    try:
        user = auth.create_user(username, password, role=role)
        task_store.audit(admin["id"], "create_user", "user", user["id"], {"username": username, "role": role})
        return f"用户 `{username}` 创建成功。", admin_refresh_users(request)
    except Exception as exc:
        raise gr.Error(f"创建用户失败: {exc}") from exc


def admin_disable_user(username: str, request: gr.Request) -> tuple[str, list[list[Any]]]:
    admin = current_admin(request)
    username = (username or "").strip()
    user = task_store.get_user_by_username(username)
    if not user:
        raise gr.Error("用户不存在。")
    if user["id"] == admin["id"]:
        raise gr.Error("不能禁用当前管理员账号。")
    task_store.set_user_active(user["id"], False)
    task_store.audit(admin["id"], "disable_user", "user", user["id"], {"username": username})
    return f"用户 `{username}` 已禁用。", admin_refresh_users(request)


def load_user_context(request: gr.Request) -> tuple[str, Any]:
    user = current_user(request)
    return (
        f"当前用户：`{user['username']}`（{user['role']}） "
        "[退出登录](/logout)",
        gr.update(visible=user.get("role") == task_store.ROLE_ADMIN),
    )


def start_batch(manifest_file, batch_mode, image_package=None, request: gr.Request = None):
    user = current_user(request)
    if manifest_file is None:
        raise gr.Error("请先上传批量任务文件。")
    if batch_mode not in {BATCH_MODE_REMOTE, BATCH_MODE_LOCAL}:
        raise gr.Error("请选择批量推理模式。")
    if batch_mode == BATCH_MODE_REMOTE and image_package is None:
        raise gr.Error("远程上传模式必须同时上传图片包 ZIP。")

    session_dir = create_session_dir(user_jobs_root(user), "qwen_image_edit_batch")
    session_id = batch_session_id(session_dir)
    session_type = "batch_remote" if batch_mode == BATCH_MODE_REMOTE else "batch_local"
    job = task_store.create_job(user["id"], session_type, session_dir, status=task_store.JOB_STATUS_QUEUED, batch_mode=batch_mode)
    metadata = initialize_session(session_dir, session_type=session_type)
    metadata["user_id"] = user["id"]
    metadata["username"] = user["username"]
    metadata["job_id"] = job["id"]
    metadata["batch_mode"] = batch_mode

    try:
        persisted_manifest, persisted_package = persist_batch_uploads(session_dir, manifest_file, image_package)
        metadata["manifest_file"] = str(persisted_manifest)
        job_update = {"manifest_file": str(persisted_manifest), "current_phase": "正在初始化"}
        if persisted_package is not None:
            metadata["uploaded_package_file"] = str(persisted_package)
            job_update["uploaded_package_file"] = str(persisted_package)
        task_store.update_job(job["id"], **job_update)

        package_root = prepare_batch_image_package(persisted_package, session_dir)
        if package_root is not None:
            metadata["extracted_package_dir"] = str(package_root)
            task_store.update_job(job["id"], extracted_package_dir=str(package_root))

        local_root = user_local_batch_root(user)
        items = parse_batch_manifest(
            persisted_manifest,
            batch_mode=batch_mode,
            package_root=package_root,
            local_batch_root=local_root,
        )
        if not items:
            raise gr.Error("批量任务文件为空。")

        for index, item in enumerate(items, start=1):
            task_store.create_job_item(
                job_id=job["id"],
                row_index=index,
                row_id=item.row_id,
                prompt=item.prompt,
                negative_prompt=item.negative_prompt,
                image_refs=item.image_refs,
                image_paths=item.image_paths,
                seed=item.seed,
                num_inference_steps=item.num_inference_steps,
                guidance_scale=item.guidance_scale,
                true_cfg_scale=item.true_cfg_scale,
            )

        task_store.update_job(
            job["id"],
            total_items=len(items),
            completed_items=0,
            success_items=0,
            failed_items=0,
            current_index=0,
            current_row_id=None,
            current_phase="等待后台任务启动",
            progress=0.0,
            download_ready=0,
            last_error=None,
        )
        write_partial_batch_results(session_dir, [])
        write_job_metadata(task_store.get_job(job["id"]))

        worker = threading.Thread(
            target=execute_batch_session,
            args=(job["id"],),
            name=f"batch-worker-{session_id}",
            daemon=True,
        )
        register_active_batch_thread(session_id, worker)
        worker.start()
        return load_batch_session_status(session_id, restore_on_load=False, default_batch_mode=batch_mode, request=request)
    except Exception as exc:
        metadata["last_error"] = str(exc)
        metadata["current_phase"] = "任务初始化失败"
        task_store.finish_job(job["id"], task_store.JOB_STATUS_FAILED, last_error=str(exc), current_phase="任务初始化失败")
        finalize_session(session_dir, metadata, status=BATCH_STATUS_FAILED)
        raise


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-Edit-2511 WebUI") as demo:
        gr.Markdown(
            "# Qwen-Image-Edit-2511 Gradio WebUI\n"
            "支持多用户登录、单次推理、多图编辑、批量任务推理和任务记录管理。"
        )
        user_status = gr.Markdown()

        with gr.Tab("单次推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    single_images = gr.File(
                        label="输入图片（支持多图）",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    single_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="请输入图像编辑指令")
                    single_negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, value=" ")
                    with gr.Row():
                        single_seed = gr.Number(label="Seed", value=0, precision=0)
                        single_steps = gr.Slider(label="推理步数", minimum=1, maximum=100, value=40, step=1)
                    with gr.Row():
                        single_guidance = gr.Slider(label="guidance_scale", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
                        single_true_cfg = gr.Slider(label="true_cfg_scale", minimum=0.1, maximum=10.0, value=4.0, step=0.1)
                    with gr.Row():
                        single_width = gr.Number(label="输出宽度", value=None, precision=0, info="留空则自动计算")
                        single_height = gr.Number(label="输出高度", value=None, precision=0, info="留空则自动计算")
                    single_button = gr.Button("开始推理", variant="primary")
                with gr.Column(scale=1):
                    single_output = gr.Image(label="生成结果", type="pil")
                    single_status = gr.Markdown(label="状态")
                    single_download = gr.File(label="下载生成图片")

            single_button.click(
                infer_single,
                inputs=[
                    single_images,
                    single_prompt,
                    single_negative_prompt,
                    single_seed,
                    single_steps,
                    single_guidance,
                    single_true_cfg,
                    single_width,
                    single_height,
                ],
                outputs=[single_output, single_status, single_download],
            )

        with gr.Tab("批量推理"):
            gr.Markdown(batch_examples_markdown())
            batch_session_state = gr.BrowserState("")
            batch_poll_timer = gr.Timer(2.0, active=False)
            batch_mode = gr.Radio(
                choices=[BATCH_MODE_REMOTE, BATCH_MODE_LOCAL],
                value=BATCH_MODE_REMOTE,
                label="批量推理模式",
            )
            batch_mode_help = gr.Markdown(batch_mode_help_text(BATCH_MODE_REMOTE))
            batch_manifest = gr.File(label="批量任务文件（CSV/JSON）", file_types=[".csv", ".json"])
            batch_image_package = gr.File(label="图片包（ZIP）", file_types=[".zip"], visible=True)
            batch_button = gr.Button("开始批量推理", variant="primary")
            batch_progress = gr.Markdown()
            batch_summary = gr.Markdown()
            batch_zip = gr.File(label="下载结果 ZIP", visible=False)

            batch_mode.change(
                update_batch_mode_ui,
                inputs=[batch_mode],
                outputs=[batch_mode_help, batch_image_package],
                queue=False,
                show_progress="hidden",
            )

            batch_button.click(
                reset_batch_outputs,
                inputs=[batch_mode],
                outputs=[
                    batch_progress,
                    batch_summary,
                    batch_zip,
                    batch_session_state,
                    batch_poll_timer,
                    batch_button,
                    batch_mode,
                    batch_mode_help,
                    batch_image_package,
                ],
                queue=False,
                show_progress="hidden",
            ).then(
                start_batch,
                inputs=[batch_manifest, batch_mode, batch_image_package],
                outputs=[
                    batch_progress,
                    batch_summary,
                    batch_zip,
                    batch_session_state,
                    batch_poll_timer,
                    batch_button,
                    batch_mode,
                    batch_mode_help,
                    batch_image_package,
                ],
            )

            batch_poll_timer.tick(
                poll_batch_status,
                inputs=[batch_session_state, batch_mode],
                outputs=[
                    batch_progress,
                    batch_summary,
                    batch_zip,
                    batch_session_state,
                    batch_poll_timer,
                    batch_button,
                    batch_mode,
                    batch_mode_help,
                    batch_image_package,
                ],
                queue=False,
                show_progress="hidden",
            )

            demo.load(
                restore_batch_session_on_load,
                inputs=[batch_session_state, batch_mode],
                outputs=[
                    batch_progress,
                    batch_summary,
                    batch_zip,
                    batch_session_state,
                    batch_poll_timer,
                    batch_button,
                    batch_mode,
                    batch_mode_help,
                    batch_image_package,
                ],
                queue=False,
                show_progress="hidden",
            )

        with gr.Tab("我的任务"):
            refresh_jobs_button = gr.Button("刷新任务列表")
            jobs_table = gr.Dataframe(
                headers=["任务ID", "用户", "类型", "状态", "进度", "创建时间", "完成时间", "下载"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
            )
            job_id_input = gr.Textbox(label="任务 ID", placeholder="从上方表格复制任务 ID")
            job_detail_button = gr.Button("查看任务详情")
            job_detail = gr.Markdown()
            job_download = gr.File(label="下载结果", visible=False)
            refresh_jobs_button.click(refresh_my_jobs, outputs=[jobs_table], queue=False, show_progress="hidden")
            job_detail_button.click(load_job_detail, inputs=[job_id_input], outputs=[job_detail, job_download], queue=False, show_progress="hidden")
            demo.load(refresh_my_jobs, outputs=[jobs_table], queue=False, show_progress="hidden")

        with gr.Tab("管理员面板"):
            with gr.Group(visible=False) as admin_panel:
                gr.Markdown("## 用户管理")
                admin_refresh_button = gr.Button("刷新用户列表")
                admin_users_table = gr.Dataframe(
                    headers=["用户ID", "用户名", "角色", "状态", "创建时间", "最近登录"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                )
                with gr.Row():
                    new_username = gr.Textbox(label="新用户名")
                    new_password = gr.Textbox(label="新用户密码", type="password")
                    new_role = gr.Radio(choices=[task_store.ROLE_USER, task_store.ROLE_ADMIN], value=task_store.ROLE_USER, label="角色")
                admin_create_button = gr.Button("添加用户", variant="primary")
                disable_username = gr.Textbox(label="要删除/禁用的用户名")
                admin_disable_button = gr.Button("删除/禁用用户")
                admin_message = gr.Markdown()
                admin_refresh_button.click(admin_refresh_users, outputs=[admin_users_table], queue=False, show_progress="hidden")
                admin_create_button.click(
                    admin_create_user,
                    inputs=[new_username, new_password, new_role],
                    outputs=[admin_message, admin_users_table],
                    queue=False,
                    show_progress="hidden",
                )
                admin_disable_button.click(
                    admin_disable_user,
                    inputs=[disable_username],
                    outputs=[admin_message, admin_users_table],
                    queue=False,
                    show_progress="hidden",
                )
                demo.load(admin_refresh_users, outputs=[admin_users_table], queue=False, show_progress="hidden")

        demo.load(load_user_context, outputs=[user_status, admin_panel], queue=False, show_progress="hidden")

    return demo


if __name__ == "__main__":
    auth.initialize_auth()
    start_cleanup_scheduler()
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=auth.authenticate)
