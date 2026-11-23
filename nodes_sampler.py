"""DiT Diffusion Sampler node for ComfyUI-Bao-Ace_Step_1_5 (Phase 1 + Phase 2).

Executes ACE-Step 1.5 flow-matching diffusion (2B or XL 4B: base, sft, turbo)
to produce 25Hz audio latents (``ACESTEP_LATENT``) on CPU across all tasks:
- ``text2music`` (Simple Mode, Custom Mode, Direct Text-to-Music)
- Reference Audio conditioning (``ACESTEP_REFERENCE_AUDIO``)
- Cover & Remix (``cover``, ``cover-nofsq``, and Flow-Edit morph overlay)
- Repaint & Edit (step-level injection, boundary blending, waveform splice handoff)
- Track Separation (``extract``)
- Multi-Track Generation (``lego`` and ``complete``)
- Vocal2BGM (``complete`` over vocal stem)
- Optional alignment tensor capture for LRC Generation and DiT Quality Scoring

Memory isolation guarantees for 12 GB VRAM:
1. Pre-encoded latents from ``AceStepVAEEncode`` are injected directly into
   conditioning without ever loading the VAE inside the Sampler.
2. ``Qwen3-Embedding-0.6B`` text/lyric encoding runs transiently and is offloaded
   back to CPU before DiT diffusion begins.
3. When ``offload_dit_to_cpu`` is enabled on the DiT Loader and free VRAM drops
   below ``vram_offload_threshold_gb``, ``DiTBlockOffloadManager`` streams DiT
   transformer blocks one-by-one between CPU and CUDA.
4. When ``unload_after_run=True`` (default), DiT and Text Encoder are purged via
   ``gc.collect()``, ``torch.cuda.empty_cache()``, and
   ``comfy.model_management.soft_empty_cache()`` in a ``finally`` block.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    from comfy.utils import ProgressBar
except ImportError:  # pragma: no cover
    ProgressBar = None

from .memory_utils import (
    AceStepDiTHandle,
    DiTBlockOffloadManager,
    get_compute_device,
    get_free_vram_gb,
    move_module_to_device_safe,
    purge_vram,
    resolve_torch_dtype,
)
from .nodes_lm import KEYSCALE_OPTIONS, TIMESIGNATURE_OPTIONS, VALID_LANGUAGES

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
COVER_DIT_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"


def _estimate_dit_resident_vram_gb(model_name: str, precision: str) -> float:
    """Conservative estimate of DiT resident weight VRAM (GB) for offload gating."""
    lower = (model_name or "").lower()
    is_xl = "xl" in lower
    if precision == "int8_dynamic":
        return 5.2 if is_xl else 2.8
    return 9.2 if is_xl else 4.9


def _check_variant_task_compatibility(dit_model: AceStepDiTHandle, handler: Any, task_cfg: Dict[str, Any]) -> None:
    """Emit actionable errors when a task requires a base/sft checkpoint instead of turbo."""
    model_lower = (dit_model.model_name or "").lower()
    is_turbo = "turbo" in model_lower or bool(getattr(getattr(handler, "config", None), "is_turbo", False))
    task_type = str(task_cfg.get("task_type", "text2music"))

    if task_cfg.get("requires_base_model", False) and is_turbo:
        raise ValueError(
            f"Task '{task_type}' (Track Separation / Multi-Track Lego / Complete / Vocal2BGM) "
            f"requires a base DiT model (e.g. 'acestep-v15-base' or 'acestep-v15-xl-base'), "
            f"but turbo checkpoint '{dit_model.model_name}' is loaded."
        )

    if task_cfg.get("requires_flow_edit_support", False):
        if is_turbo or not hasattr(handler.model, "flowedit_generate_audio"):
            raise ValueError(
                f"Flow-Edit morph overlay requires a base or SFT DiT model exposing "
                f"'flowedit_generate_audio' (e.g. 'acestep-v15-xl-sft', 'acestep-v15-xl-base', "
                f"'acestep-v15-sft', or 'acestep-v15-base'), but '{dit_model.model_name}' does not support it."
            )


def _extract_alignment_tensors_cpu(outputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detach intermediate cross-attention conditioning tensors to CPU for LRC & Quality Scoring."""
    enc_hs = outputs.get("encoder_hidden_states")
    enc_am = outputs.get("encoder_attention_mask")
    ctx_lat = outputs.get("context_latents")
    lyric_ids = outputs.get("lyric_token_ids") or outputs.get("lyric_token_idss")
    if enc_hs is None or enc_am is None or ctx_lat is None or lyric_ids is None:
        return None
    return {
        "encoder_hidden_states": enc_hs.detach().cpu(),
        "encoder_attention_mask": enc_am.detach().cpu(),
        "context_latents": ctx_lat.detach().cpu(),
        "lyric_token_ids": lyric_ids.detach().cpu() if isinstance(lyric_ids, torch.Tensor) else lyric_ids,
    }


class AceStepSampler:
    """Run ACE-Step 1.5 DiT diffusion to generate 25Hz audio latents."""

    CATEGORY = "ACE-Step 1.5/Sampling"
    RETURN_TYPES = ("ACESTEP_LATENT", "STRING")
    RETURN_NAMES = ("latent", "generation_info")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_model": (
                    "ACESTEP_DIT",
                    {"tooltip": "DiT handle from ACE-Step 1.5 DiT Loader."},
                ),
                "inference_steps": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": (
                            "Number of diffusion denoising steps. "
                            "Recommended: 8 for turbo variants, 32-60 for base/sft variants."
                        ),
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": (
                            "Classifier-Free Guidance scale (effective on base/sft models; "
                            "set to 1.0 on turbo models which run CFG-free)."
                        ),
                    },
                ),
                "shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.5,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Flow-matching timestep shift factor (typically 3.0 for turbo, 1.0-3.0 for sft/base).",
                    },
                ),
                "infer_method": (
                    ["ode", "sde"],
                    {
                        "default": "ode",
                        "tooltip": "Diffusion solver formulation: deterministic 'ode' or stochastic 'sde'.",
                    },
                ),
                "sampler_mode": (
                    ["euler", "heun"],
                    {
                        "default": "euler",
                        "tooltip": "Numerical integrator for diffusion steps ('euler' is faster; 'heun' is 2nd-order).",
                    },
                ),
                "use_adg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable Adaptive Dual Guidance (ADG) on CFG-capable base/sft models.",
                    },
                ),
                "cfg_interval_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Normalized timestep ratio where CFG guidance begins.",
                    },
                ),
                "cfg_interval_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Normalized timestep ratio where CFG guidance ends.",
                    },
                ),
                "audio_cover_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Influence of 5Hz LM semantic codes or Cover source structure "
                            "(overridden by AceStepCoverTask when connected)."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": -1,
                        "max": 2**32 - 1,
                        "tooltip": "Diffusion noise seed (-1 for random).",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of tracks to synthesize in parallel (keep 1 for 12 GB VRAM with XL 4B).",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Mandatory for 12 GB VRAM: unload DiT and Qwen3-Embedding-0.6B "
                            "and purge CUDA/ComfyUI caches before VAE decoding."
                        ),
                    },
                ),
            },
            "optional": {
                "lm_plan": (
                    "ACESTEP_PLAN",
                    {
                        "tooltip": (
                            "Optional plan from ACE-Step 1.5 LM Planner (caption, lyrics, metadata, 5Hz codes)."
                        ),
                    },
                ),
                "task_config": (
                    "ACESTEP_TASK_CONFIG",
                    {
                        "tooltip": (
                            "Optional task configuration from Cover/Flow-Edit, Repaint & Edit, "
                            "Track Separation (Extract), Multi-Track (Lego/Complete), or Vocal2BGM."
                        ),
                    },
                ),
                "reference_audio": (
                    "ACESTEP_REFERENCE_AUDIO",
                    {
                        "tooltip": (
                            "Optional pre-encoded style/timbre reference from ACE-Step 1.5 Reference Audio."
                        ),
                    },
                ),
                "metadata": (
                    "ACESTEP_METADATA",
                    {
                        "tooltip": (
                            "Optional structured metadata from Metadata Control, Query Rewriter, or Audio Understanding."
                        ),
                    },
                ),
                "caption_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Direct style tags / caption (overrides lm_plan and metadata when non-empty).",
                    },
                ),
                "lyrics_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Direct structured lyrics ([Verse], [Chorus], ...). Overrides lm_plan/metadata when non-empty.",
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force instrumental generation.",
                    },
                ),
                "vocal_language": (
                    VALID_LANGUAGES,
                    {
                        "default": "unknown",
                        "tooltip": "Vocal language override ('unknown' preserves lm_plan/metadata).",
                    },
                ),
                "bpm_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "step": 1,
                        "tooltip": "Override BPM (30-300). 0 keeps value from lm_plan/metadata.",
                    },
                ),
                "keyscale_override": (
                    KEYSCALE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Override musical keyscale ('auto' keeps lm_plan/metadata value).",
                    },
                ),
                "timesignature_override": (
                    TIMESIGNATURE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Override time signature ('auto' keeps lm_plan/metadata value).",
                    },
                ),
                "duration_override": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "Override duration in seconds (10-600s). -1 uses task source audio, lm_plan, or 60s.",
                    },
                ),
                "audio_codes_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional raw 5Hz semantic audio codes string overriding lm_plan.",
                    },
                ),
                "capture_alignment_data": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Store CPU-offloaded conditioning tensors inside ACESTEP_LATENT so downstream "
                            "LRC Generation and DiT Quality Scoring nodes can run without re-sampling."
                        ),
                    },
                ),
                "latent_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Optional post-diffusion latent additive shift (default 0.0).",
                    },
                ),
                "latent_rescale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Optional post-diffusion latent multiplicative rescale (default 1.0).",
                    },
                ),
            },
        }

    def sample(
        self,
        dit_model: AceStepDiTHandle,
        inference_steps: int,
        guidance_scale: float,
        shift: float,
        infer_method: str,
        sampler_mode: str,
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        audio_cover_strength: float,
        seed: int,
        batch_size: int,
        unload_after_run: bool,
        lm_plan: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        reference_audio: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        caption_override: str = "",
        lyrics_override: str = "",
        instrumental: bool = False,
        vocal_language: str = "unknown",
        bpm_override: int = 0,
        keyscale_override: str = "auto",
        timesignature_override: str = "auto",
        duration_override: float = -1.0,
        audio_codes_override: str = "",
        capture_alignment_data: bool = False,
        latent_shift: float = 0.0,
        latent_rescale: float = 1.0,
    ) -> Tuple[Dict[str, Any], str]:
        t0 = time.time()
        pbar = ProgressBar(4) if ProgressBar is not None else None

        task_cfg = task_config if isinstance(task_config, dict) else {}
        plan_meta = (lm_plan or {}).get("metadata", {}) if isinstance(lm_plan, dict) else {}
        ext_meta = metadata if isinstance(metadata, dict) else {}

        # Merge caption: override > metadata > lm_plan
        caption = (
            caption_override.strip()
            or str(ext_meta.get("caption", "")).strip()
            or str((lm_plan or {}).get("caption", "")).strip()
        )
        if not caption:
            if task_cfg.get("task_type") in ("extract", "lego", "complete"):
                # Stem extraction / completion can operate with minimal caption if omitted
                caption = f"{task_cfg.get('track_name', 'music')} track"
            else:
                raise ValueError(
                    "ACE-Step 1.5 Sampler requires a caption via lm_plan, metadata, or caption_override."
                )

        is_instrumental = (
            instrumental
            or bool(ext_meta.get("instrumental", False))
            or bool((lm_plan or {}).get("instrumental", False))
        )
        if lyrics_override.strip():
            lyrics = lyrics_override.strip()
        elif is_instrumental:
            lyrics = "[Instrumental]"
        else:
            lyrics = str(ext_meta.get("lyrics", "")).strip() or str((lm_plan or {}).get("lyrics", "")).strip()

        resolved_language = (
            vocal_language
            if (vocal_language and vocal_language != "unknown")
            else (
                str(ext_meta.get("vocal_language", "")).strip()
                or str(plan_meta.get("vocal_language", "unknown"))
            )
        )
        if is_instrumental or not resolved_language:
            resolved_language = "unknown"

        resolved_bpm: Optional[int] = (
            bpm_override
            if 30 <= bpm_override <= 300
            else (ext_meta.get("bpm") if ext_meta.get("bpm") is not None else plan_meta.get("bpm"))
        )
        resolved_keyscale = (
            keyscale_override
            if keyscale_override != "auto"
            else (str(ext_meta.get("keyscale", "")).strip() or str(plan_meta.get("keyscale", "")))
        )
        resolved_timesig = (
            timesignature_override
            if timesignature_override != "auto"
            else (str(ext_meta.get("timesignature", "")).strip() or str(plan_meta.get("timesignature", "")))
        )

        # Duration priority: duration_override > source audio duration > ext_meta > lm_plan > 60s
        src_audio_bundle = task_cfg.get("source_audio_latent")
        if duration_override > 0:
            resolved_duration = max(10.0, min(600.0, float(duration_override)))
        elif isinstance(src_audio_bundle, dict) and float(src_audio_bundle.get("duration", 0.0)) > 0:
            resolved_duration = max(1.0, min(600.0, float(src_audio_bundle["duration"])))
        elif ext_meta.get("duration") and float(ext_meta["duration"]) > 0:
            resolved_duration = max(10.0, min(600.0, float(ext_meta["duration"])))
        elif plan_meta.get("duration") and float(plan_meta["duration"]) > 0:
            resolved_duration = max(10.0, min(600.0, float(plan_meta["duration"])))
        else:
            resolved_duration = 60.0

        actual_batch_size = max(1, int(batch_size))

        # Resolve 5Hz audio codes: explicit override > task_config source codes > lm_plan
        audio_codes_input: Union[str, List[str]] = ""
        if audio_codes_override.strip():
            audio_codes_input = audio_codes_override.strip()
        elif task_cfg.get("source_audio_codes"):
            audio_codes_input = str(task_cfg["source_audio_codes"]).strip()
        elif isinstance(lm_plan, dict) and lm_plan.get("audio_codes"):
            plan_codes = lm_plan["audio_codes"]
            if isinstance(plan_codes, list):
                cleaned = [str(c or "") for c in plan_codes]
                if len(cleaned) >= actual_batch_size:
                    audio_codes_input = cleaned[:actual_batch_size]
                elif cleaned:
                    audio_codes_input = (cleaned * actual_batch_size)[:actual_batch_size]
            else:
                audio_codes_input = str(plan_codes)

        base_seed = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
        seed_list = [(base_seed + i) % (2**32) for i in range(actual_batch_size)]

        block_offload_mgr: Optional[DiTBlockOffloadManager] = None
        block_offload_used = False
        orig_load_model_context = None
        orig_infer_refer_latent = None
        orig_tiled_encode = None
        active_handler = None
        alignment_data_cpu: Optional[Dict[str, Any]] = None

        try:
            handler = dit_model.ensure_loaded()
            active_handler = handler
            exec_device = get_compute_device("cuda")
            compute_dtype = resolve_torch_dtype(dit_model.precision, exec_device)
            if pbar is not None:
                pbar.update(1)

            if task_cfg:
                _check_variant_task_compatibility(dit_model, handler, task_cfg)

            has_codes = (
                any(c.strip() for c in audio_codes_input)
                if isinstance(audio_codes_input, list)
                else bool(audio_codes_input.strip())
            )

            # Resolve task_type, instruction, and task-specific parameters
            if task_cfg:
                task_type = str(task_cfg.get("task_type", "text2music"))
                instruction = str(task_cfg.get("instruction", DEFAULT_DIT_INSTRUCTION))
                effective_cover_strength = float(task_cfg.get("audio_cover_strength", audio_cover_strength))
                effective_cover_noise = float(task_cfg.get("cover_noise_strength", 0.0))
                repaint_start = float(task_cfg.get("repainting_start", 0.0))
                repaint_end = task_cfg.get("repainting_end", None)
                repaint_crossfade_frames = int(task_cfg.get("repaint_crossfade_frames", 10))
                repaint_injection_ratio = float(task_cfg.get("repaint_injection_ratio", 0.5))
                chunk_mask_mode = str(task_cfg.get("chunk_mask_mode", "auto"))
                flow_edit_morph = bool(task_cfg.get("flow_edit_morph", False))
                flow_edit_source_caption = str(task_cfg.get("flow_edit_source_caption", ""))
                flow_edit_source_lyrics = str(task_cfg.get("flow_edit_source_lyrics", ""))
                flow_edit_n_min = float(task_cfg.get("flow_edit_n_min", 0.0))
                flow_edit_n_max = float(task_cfg.get("flow_edit_n_max", 1.0))
                flow_edit_n_avg = int(task_cfg.get("flow_edit_n_avg", 1))
            else:
                task_type = "cover" if has_codes else "text2music"
                instruction = COVER_DIT_INSTRUCTION if has_codes else DEFAULT_DIT_INSTRUCTION
                effective_cover_strength = audio_cover_strength if has_codes else 1.0
                effective_cover_noise = 0.0
                repaint_start = 0.0
                repaint_end = None
                repaint_crossfade_frames = 10
                repaint_injection_ratio = 0.5
                chunk_mask_mode = "auto"
                flow_edit_morph = False
                flow_edit_source_caption = ""
                flow_edit_source_lyrics = ""
                flow_edit_n_min = 0.0
                flow_edit_n_max = 1.0
                flow_edit_n_avg = 1

            sample_rate = getattr(handler, "sample_rate", 48000)

            # Inject pre-encoded Reference Audio without loading VAE
            refer_audios = [[torch.zeros(2, 30 * sample_rate)] for _ in range(actual_batch_size)]
            if isinstance(reference_audio, dict) and isinstance(reference_audio.get("latents"), torch.Tensor):
                ref_lat_cpu = reference_audio["latents"].detach().cpu()  # [1, T_ref, 64]
                if ref_lat_cpu.dim() == 2:
                    ref_lat_cpu = ref_lat_cpu.unsqueeze(0)
                orig_infer_refer_latent = handler.infer_refer_latent

                def _preencoded_infer_refer_latent(_refer_audioss):
                    ref_gpu = ref_lat_cpu.to(device=exec_device, dtype=compute_dtype)
                    if ref_gpu.shape[0] == 1 and actual_batch_size > 1:
                        ref_gpu = ref_gpu.repeat(actual_batch_size, 1, 1)
                    order_mask = torch.arange(actual_batch_size, device=exec_device, dtype=torch.long)
                    return ref_gpu, order_mask

                handler.infer_refer_latent = _preencoded_infer_refer_latent

            # Inject pre-encoded Source Audio latent without loading VAE
            processed_src_audio: Optional[torch.Tensor] = None
            source_repaint_latents: Optional[torch.Tensor] = None
            if isinstance(src_audio_bundle, dict) and isinstance(src_audio_bundle.get("latents"), torch.Tensor):
                src_lat_btc = src_audio_bundle["latents"].detach().cpu()  # [1, T, 64]
                src_lat_bct = src_audio_bundle.get("latents_bct")
                if src_lat_bct is None:
                    src_lat_bct = src_lat_btc.transpose(1, 2).contiguous()
                src_wav_cpu = src_audio_bundle.get("waveform")
                if isinstance(src_wav_cpu, torch.Tensor):
                    processed_src_audio = src_wav_cpu[0] if src_wav_cpu.dim() == 3 else src_wav_cpu
                else:
                    num_samples = int(round(src_lat_btc.shape[1] * 1920))
                    # Non-zero sentinel so is_silence() does not discard pre-encoded latents
                    processed_src_audio = torch.full((2, max(1920, num_samples)), 1e-3, dtype=torch.float32)

                source_repaint_latents = src_lat_btc
                orig_tiled_encode = handler.tiled_encode

                def _preencoded_tiled_encode(audio, chunk_size=None, overlap=None, offload_latent_to_cpu=True):
                    _ = audio, chunk_size, overlap, offload_latent_to_cpu
                    return src_lat_bct.clone()

                handler.tiled_encode = _preencoded_tiled_encode

            service_inputs = handler._prepare_generate_music_service_inputs(
                actual_batch_size=actual_batch_size,
                processed_src_audio=processed_src_audio,
                audio_duration=resolved_duration,
                captions=caption,
                global_caption="",
                lyrics=lyrics,
                vocal_language=resolved_language,
                instruction=instruction,
                bpm=resolved_bpm,
                key_scale=resolved_keyscale,
                time_signature=resolved_timesig,
                task_type=task_type,
                audio_code_string=audio_codes_input,
                repainting_start=repaint_start,
                repainting_end=repaint_end,
                chunk_mask_mode=chunk_mask_mode,
            )
            if pbar is not None:
                pbar.update(1)

            # Configure DiT device placement / block-level offloading for 12 GB VRAM safety
            if exec_device.type == "cuda":
                free_before = get_free_vram_gb(exec_device)
                est_dit_gb = _estimate_dit_resident_vram_gb(dit_model.model_name, dit_model.precision)
                projected_free_after_load = free_before - est_dit_gb
                needs_block_offload = (
                    dit_model.offload_dit_to_cpu
                    and projected_free_after_load < dit_model.vram_offload_threshold_gb
                )
                if needs_block_offload:
                    block_offload_mgr = DiTBlockOffloadManager(
                        model=handler.model,
                        exec_device=exec_device,
                        compute_dtype=compute_dtype,
                        vram_threshold_gb=dit_model.vram_offload_threshold_gb,
                        force_block_offload=True,
                    )
                    block_offload_used = block_offload_mgr.attach_if_needed()
                    if block_offload_used:
                        orig_load_model_context = handler._load_model_context

                        @contextmanager
                        def _block_offload_aware_context(model_name: str):
                            if model_name in ("model", "vae"):
                                yield
                                return
                            with orig_load_model_context(model_name):
                                yield

                        handler._load_model_context = _block_offload_aware_context
                        handler.silence_latent = handler.silence_latent.to(exec_device, dtype=compute_dtype)
                else:
                    handler.offload_dit_to_cpu = dit_model.offload_dit_to_cpu

            logger.info(
                "[ACE-Step 1.5] Sampling DiT '%s' (task=%s, precision=%s, steps=%d, cfg=%.1f, "
                "duration=%.1fs, ref=%s, flow_edit=%s, block_offload=%s)...",
                dit_model.model_name,
                task_type,
                dit_model.precision,
                inference_steps,
                guidance_scale,
                resolved_duration,
                reference_audio is not None,
                flow_edit_morph,
                block_offload_used,
            )

            with torch.inference_mode():
                outputs = handler.service_generate(
                    captions=service_inputs["captions_batch"],
                    global_captions=service_inputs["global_captions_batch"],
                    lyrics=service_inputs["lyrics_batch"],
                    metas=service_inputs["metas_batch"],
                    vocal_languages=service_inputs["vocal_languages_batch"],
                    refer_audios=refer_audios,
                    target_wavs=service_inputs["target_wavs_tensor"],
                    infer_steps=int(inference_steps),
                    guidance_scale=float(guidance_scale),
                    seed=seed_list,
                    return_intermediate=bool(capture_alignment_data),
                    repainting_start=service_inputs["repainting_start_batch"],
                    repainting_end=service_inputs["repainting_end_batch"],
                    instructions=service_inputs["instructions_batch"],
                    audio_cover_strength=float(effective_cover_strength),
                    cover_noise_strength=float(effective_cover_noise),
                    use_adg=bool(use_adg),
                    cfg_interval_start=float(cfg_interval_start),
                    cfg_interval_end=float(cfg_interval_end),
                    shift=float(shift),
                    audio_code_hints=service_inputs["audio_code_hints_batch"],
                    infer_method=infer_method,
                    chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
                    repaint_crossfade_frames=repaint_crossfade_frames,
                    repaint_injection_ratio=repaint_injection_ratio,
                    source_repaint_latents=source_repaint_latents if task_type == "repaint" else None,
                    sampler_mode=sampler_mode,
                    task_type=task_type,
                    flow_edit_morph=flow_edit_morph,
                    flow_edit_source_caption=flow_edit_source_caption,
                    flow_edit_source_lyrics=flow_edit_source_lyrics,
                    flow_edit_n_min=flow_edit_n_min,
                    flow_edit_n_max=flow_edit_n_max,
                    flow_edit_n_avg=flow_edit_n_avg,
                )

            if pbar is not None:
                pbar.update(1)

            pred_latents = outputs.get("target_latents")
            if pred_latents is None:
                raise RuntimeError("DiT service_generate returned no 'target_latents'.")

            if capture_alignment_data:
                alignment_data_cpu = _extract_alignment_tensors_cpu(outputs)

            pred_latents = pred_latents.detach().cpu()
            if latent_shift != 0.0 or latent_rescale != 1.0:
                pred_latents = pred_latents * float(latent_rescale) + float(latent_shift)

            del outputs
            if pbar is not None:
                pbar.update(1)
        finally:
            if active_handler is not None:
                if orig_load_model_context is not None:
                    active_handler._load_model_context = orig_load_model_context
                if orig_infer_refer_latent is not None:
                    active_handler.infer_refer_latent = orig_infer_refer_latent
                if orig_tiled_encode is not None:
                    active_handler.tiled_encode = orig_tiled_encode
            if block_offload_mgr is not None:
                block_offload_mgr.detach()
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading DiT & Text Encoder after sampling (unload_after_run=True).")
                dit_model.unload(release_reference=True)
            else:
                if dit_model.handler is not None and getattr(dit_model.handler, "text_encoder", None) is not None:
                    move_module_to_device_safe(dit_model.handler.text_encoder, "cpu")
                purge_vram(aggressive=False)

        elapsed = time.time() - t0
        repaint_splice_payload: Optional[Dict[str, Any]] = None
        if (
            task_cfg.get("task_type") == "repaint"
            and task_cfg.get("enable_waveform_splice", False)
            and isinstance(src_audio_bundle, dict)
            and src_audio_bundle.get("waveform") is not None
        ):
            repaint_splice_payload = {
                "enabled": True,
                "src_waveform": src_audio_bundle["waveform"].detach().cpu(),
                "repainting_start": float(task_cfg.get("repainting_start", 0.0)),
                "repainting_end": float(task_cfg.get("repainting_end", -1.0)),
                "wav_crossfade_sec": float(task_cfg.get("wav_crossfade_sec", 0.025)),
            }

        first_codes = (
            audio_codes_input[0]
            if isinstance(audio_codes_input, list) and audio_codes_input
            else (audio_codes_input if isinstance(audio_codes_input, str) else "")
        )

        latent_payload: Dict[str, Any] = {
            "samples": pred_latents,  # [batch, frames_25hz, 64] on CPU
            "latent_rate_hz": 25,
            "sample_rate": 48000,
            "duration": resolved_duration,
            "batch_size": actual_batch_size,
            "seeds": seed_list,
            "dit_model": dit_model.model_name,
            "precision": dit_model.precision,
            "task_type": task_type,
            "audio_codes": first_codes,
            "alignment_data": alignment_data_cpu,
            "repaint_splice": repaint_splice_payload,
            "metadata": {
                "caption": caption,
                "lyrics": lyrics,
                "bpm": resolved_bpm,
                "keyscale": resolved_keyscale,
                "timesignature": resolved_timesig,
                "vocal_language": resolved_language,
                "instrumental": is_instrumental,
                "used_lm_codes": has_codes,
                "used_reference_audio": reference_audio is not None,
                "flow_edit_morph": flow_edit_morph,
            },
        }

        info = (
            f"Task={task_type} | DiT={dit_model.model_name} ({dit_model.precision}, quant={dit_model.quant_backend}) | "
            f"shape={tuple(pred_latents.shape)} | duration={resolved_duration:.1f}s | "
            f"steps={inference_steps} | cfg={guidance_scale} | shift={shift} | "
            f"block_offload={block_offload_used} | alignment_captured={alignment_data_cpu is not None} | "
            f"elapsed={elapsed:.2f}s"
        )
        logger.info("[ACE-Step 1.5] %s", info)
        return (latent_payload, info)


NODE_CLASS_MAPPINGS = {
    "AceStepSampler": AceStepSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSampler": "ACE-Step 1.5 DiT Sampler (25Hz Latents)",
}
