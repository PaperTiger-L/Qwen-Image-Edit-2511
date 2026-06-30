import csv
import functools
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional

import gradio as gr
import torch
from PIL import Image
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
OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        file_path = Path(getattr(file, "name", file))
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


def save_single_result(image: Image.Image, image_paths: List[Path], seed: int) -> str:
    input_stem = get_primary_input_stem(image_paths)
    file_name = f"{input_stem}_seed{seed}.png"
    output_path = SINGLE_OUTPUT_DIR / file_name
    image.save(output_path)
    return str(output_path)


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
):
    image_paths = parse_uploaded_images(images)
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
    saved_path = save_single_result(result, image_paths, seed)
    resolution_text = "自动" if not width and not height else f"{int(width)} × {int(height)}"
    status = (
        f"推理完成\n\n"
        f"- device: `{_DEVICE}`\n"
        f"- dtype: `{_DTYPE}`\n"
        f"- device_map: `{_DEVICE_MAP_INFO}`\n"
        f"- attention_slicing: `{os.getenv('ENABLE_ATTENTION_SLICING', '0')}`\n"
        f"- cpu_offload: `{os.getenv('ENABLE_CPU_OFFLOAD', '0')}`\n"
        f"- 输入图片数: {len(image_paths)}\n"
        f"- 输出分辨率: `{resolution_text}`\n"
        f"- 输出文件: `{saved_path}`"
    )
    return result, status, saved_path


def load_batch_manifest_rows(file_obj) -> tuple[Path, List[dict[str, Any]]]:
    if file_obj is None:
        raise gr.Error("请上传批量任务文件（CSV 或 JSON）。")

    file_path = Path(getattr(file_obj, "name", file_obj)).resolve()
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


def resolve_batch_image_path(image_ref: str, manifest_dir: Path, package_root: Optional[Path] = None) -> Path:
    image_ref = image_ref.strip()
    if not image_ref:
        raise gr.Error("批量任务中的 images 字段不能为空。")

    ref_path = Path(image_ref)
    if package_root is not None:
        if ref_path.is_absolute():
            raise gr.Error(
                "检测到已上传图片包，但任务文件中的 images 使用了绝对路径。"
                "请使用 generate_batch_manifest.py 的 --image-path-mode package-relative 重新生成任务文件。"
            )
        normalized_ref = PurePosixPath(image_ref)
        if normalized_ref.is_absolute() or ".." in normalized_ref.parts:
            raise gr.Error(f"图片包模式下不允许越界路径: {image_ref}")
        resolved_path = (package_root / Path(*normalized_ref.parts)).resolve()
        ensure_relative_to_root(resolved_path, package_root)
        return resolved_path

    candidate_paths = []
    if ref_path.is_absolute():
        candidate_paths.append(ref_path)
    else:
        candidate_paths.append((manifest_dir / ref_path).resolve())
        fallback_path = (MODEL_DIR / ref_path).resolve()
        if fallback_path not in candidate_paths:
            candidate_paths.append(fallback_path)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def prepare_batch_image_package(package_file, session_dir: Path) -> Optional[Path]:
    if package_file is None:
        return None

    package_path = Path(getattr(package_file, "name", package_file)).resolve()
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


def parse_batch_manifest(file_obj, package_root: Optional[Path] = None) -> List[BatchItem]:
    manifest_path, rows = load_batch_manifest_rows(file_obj)
    manifest_dir = manifest_path.parent

    items: List[BatchItem] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or index)
        prompt = str(row.get("prompt") or "").strip()
        negative_prompt = str(row.get("negative_prompt") or " ")
        image_refs = parse_row_image_refs(row)
        image_paths = [resolve_batch_image_path(image_ref, manifest_dir, package_root=package_root) for image_ref in image_refs]
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
1,A silver robot standing in a flower field., ,examples/input1.png,0,40,1.0,4.0
2,Merge the two people into one travel photo., ,examples/a.png|examples/b.png,42,40,1.0,4.0"""
    json_example = [
        {
            "id": "1",
            "prompt": "A silver robot standing in a flower field.",
            "negative_prompt": " ",
            "images": ["examples/input1.png"],
            "seed": 0,
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
    ]
    return (
        "### 批量任务文件格式\n"
        "支持两种模式：\n"
        "- **本地模式**：只上传 `CSV/JSON`，`images` 可写绝对路径；相对路径优先按任务文件所在目录解析，再兼容当前项目目录。\n"
        "- **远程上传模式**：上传 `CSV/JSON` + `ZIP` 图片包，`images` 必须写成相对 `ZIP` 根目录的路径。\n"
        "多图输入在 CSV 中使用 `|` 分隔。远程上传模式建议用 `generate_batch_manifest.py --image-path-mode package-relative` 生成任务文件。\n\n"
        "**CSV 示例**\n"
        f"```csv\n{csv_example}\n```\n"
        "**JSON 示例**\n"
        f"```json\n{json.dumps(json_example, ensure_ascii=False, indent=2)}\n```"
    )


def run_batch(manifest_file, image_package=None, progress=gr.Progress(track_tqdm=False)):
    session_dir = Path(tempfile.mkdtemp(prefix="qwen_image_edit_batch_", dir=BATCH_OUTPUT_DIR))
    package_root = prepare_batch_image_package(image_package, session_dir)
    items = parse_batch_manifest(manifest_file, package_root=package_root)
    if not items:
        raise gr.Error("批量任务文件为空。")

    rows = []
    gallery = []

    for idx, item in enumerate(items, start=1):
        progress((idx - 1) / len(items), desc=f"处理中 {idx}/{len(items)}")
        try:
            result = run_generation(
                image_paths=item.image_paths,
                prompt=item.prompt,
                negative_prompt=item.negative_prompt,
                seed=item.seed,
                num_inference_steps=item.num_inference_steps,
                guidance_scale=item.guidance_scale,
                true_cfg_scale=item.true_cfg_scale,
            )
            input_stem = get_primary_input_stem(item.image_paths)
            file_name = f"{idx:03d}_{input_stem}.png"
            output_path = session_dir / file_name
            result.save(output_path)
            rows.append(
                {
                    "id": item.row_id,
                    "prompt": item.prompt,
                    "negative_prompt": item.negative_prompt,
                    "images": "|".join(item.image_refs),
                    "seed": item.seed,
                    "num_inference_steps": item.num_inference_steps,
                    "guidance_scale": item.guidance_scale,
                    "true_cfg_scale": item.true_cfg_scale,
                    "status": "success",
                    "output_image": str(output_path),
                    "error": "",
                }
            )
            gallery.append((str(output_path), f"{item.row_id}: {item.prompt[:60]}"))
        except Exception as exc:
            rows.append(
                {
                    "id": item.row_id,
                    "prompt": item.prompt,
                    "negative_prompt": item.negative_prompt,
                    "images": "|".join(item.image_refs),
                    "seed": item.seed,
                    "num_inference_steps": item.num_inference_steps,
                    "guidance_scale": item.guidance_scale,
                    "true_cfg_scale": item.true_cfg_scale,
                    "status": "failed",
                    "output_image": "",
                    "error": str(exc),
                }
            )

    progress(1.0, desc="正在整理结果")
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

    preview_rows = [
        [
            row["id"],
            row["status"],
            row["prompt"],
            row["output_image"],
            row["error"],
        ]
        for row in rows
    ]
    success_count = sum(1 for row in rows if row["status"] == "success")
    summary = (
        f"批量推理完成\n\n"
        f"- 总任务数: {len(rows)}\n"
        f"- 成功: {success_count}\n"
        f"- 失败: {len(rows) - success_count}\n"
        f"- 结果目录: `{session_dir}`"
    )
    return preview_rows, gallery, summary, str(csv_path), str(json_path), str(zip_path)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-Edit-2511 WebUI") as demo:
        gr.Markdown(
            "# Qwen-Image-Edit-2511 Gradio WebUI\n"
            "支持单次推理、多图编辑和批量任务推理。模型从当前目录本地加载。"
        )

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
            batch_manifest = gr.File(label="批量任务文件（CSV/JSON）", file_types=[".csv", ".json"])
            batch_image_package = gr.File(label="图片包（ZIP，可选）", file_types=[".zip"])
            batch_button = gr.Button("开始批量推理", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(
                headers=["id", "status", "prompt", "output_image", "error"],
                datatype=["str", "str", "str", "str", "str"],
                label="批量结果概览",
                interactive=False,
                wrap=True,
            )
            batch_gallery = gr.Gallery(label="成功结果预览", columns=3, height="auto")
            with gr.Row():
                batch_csv = gr.File(label="导出 CSV")
                batch_json = gr.File(label="导出 JSON")
                batch_zip = gr.File(label="导出全部结果 ZIP")

            batch_button.click(
                run_batch,
                inputs=[batch_manifest, batch_image_package],
                outputs=[batch_table, batch_gallery, batch_summary, batch_csv, batch_json, batch_zip],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
