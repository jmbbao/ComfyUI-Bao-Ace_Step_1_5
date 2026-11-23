"""VAE Encode & Decode nodes for ComfyUI-Bao-Ace_Step_1_5.

Provides strictly isolated 48 kHz <-> 25 Hz audio latent conversion using
``AutoencoderOobleck``:

- ``AceStepVAEEncode``: Converts ComfyUI ``AUDIO`` (from ``LoadAudio`` or any
  audio node) into a CPU-resident ``ACESTEP_AUDIO_LATENT`` bundle (25 Hz latents
  + resampled 48 kHz stereo waveform reference) and immediately purges the VAE
  from VRAM so the VAE never coexists with DiT XL 4B or LM 4B.
- ``AceStepVAEDecode``: Decodes 25 Hz ``ACESTEP_LATENT`` tensors into ComfyUI
  native ``AUDIO`` via overlap-discard tiled decoding, with optional CPU-side
  Repaint waveform splicing (preserving bit-exact original audio outside the
  repainted interval with smooth linear edge crossfades) and automatic CPU OOM
  fallback.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from comfy.utils import ProgressBar
except ImportError:  # pragma: no cover
    ProgressBar = None

from .memory_utils import (
    AceStepVAEHandle,
    move_module_to_device_safe,
    purge_vram,
)

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")

TARGET_SAMPLE_RATE = 48000
LATENT_FRAME_HOP = 1920  # 48000 Hz / 25 Hz


def _normalize_input_audio_stereo_48k(
    audio: Dict[str, Any],
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize_peak: bool = True,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = -1.0,
) -> Tuple[torch.Tensor, float]:
    """Normalize ComfyUI ``AUDIO`` dict to ``[1, 2, samples]`` float32 CPU at 48 kHz."""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Invalid ComfyUI AUDIO input: expected dict with 'waveform' and 'sample_rate'.")

    wav = audio["waveform"]
    sr = int(audio.get("sample_rate", target_sr))
    if not isinstance(wav, torch.Tensor):
        wav = torch.as_tensor(wav)

    wav = wav.detach().to(device="cpu", dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)
    elif wav.dim() > 3:
        wav = wav.reshape(wav.shape[0], wav.shape[1], -1)

    # Use first batch element for single-track reference/source conditioning
    wav = wav[:1]
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    elif wav.shape[1] > 2:
        wav = wav[:, :2, :]

    # Optional temporal trimming in source sample rate
    total_src_samples = wav.shape[-1]
    start_idx = max(0, int(round(max(0.0, trim_start_sec) * sr)))
    end_idx = total_src_samples if trim_end_sec <= 0 else min(total_src_samples, int(round(trim_end_sec * sr)))
    if end_idx > start_idx:
        wav = wav[..., start_idx:end_idx]

    # Resample to 48 kHz if needed
    if sr != target_sr and wav.shape[-1] > 0:
        resampled = None
        try:
            import torchaudio
            resampled = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        except Exception:
            pass

        if resampled is None:
            try:
                from math import gcd
                import numpy as np
                from scipy.signal import resample_poly

                g = gcd(target_sr, sr)
                up, down = target_sr // g, sr // g
                arr = wav.numpy()
                out_np = resample_poly(arr, up, down, axis=-1).astype(np.float32)
                resampled = torch.from_numpy(out_np)
            except Exception:
                new_len = max(1, int(round(wav.shape[-1] * (target_sr / float(sr)))))
                resampled = F.interpolate(wav, size=new_len, mode="linear", align_corners=False)
        wav = resampled

    wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
    if normalize_peak:
        peak = float(wav.abs().max().item()) if wav.numel() > 0 else 0.0
        if peak > 1.0:
            wav = wav / peak

    duration_sec = wav.shape[-1] / float(target_sr)
    return wav.contiguous(), duration_sec


def _normalize_latents_bct(samples: torch.Tensor) -> torch.Tensor:
    """Ensure latents are shaped ``[batch, channels=64, frames]`` for AutoencoderOobleck."""
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    if samples.dim() != 3:
        raise ValueError(f"Expected 3D latent tensor [B, T, C] or [B, C, T], got shape {tuple(samples.shape)}")

    # DiT outputs [B, frames, 64]; AutoencoderOobleck expects [B, 64, frames].
    if samples.shape[1] != 64 and samples.shape[2] == 64:
        samples = samples.transpose(1, 2).contiguous()
    return samples


def _normalize_waveform_comfy(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize decoded waveform to ComfyUI native ``AUDIO`` tensor ``[B, 2, T]`` float32 CPU."""
    wav = waveform.detach().to(device="cpu", dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    elif wav.shape[1] > 2:
        wav = wav[:, :2, :]
    wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
    return wav.contiguous()


def _build_waveform_crossfade_mask(
    total_samples: int,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-sample mask: 1.0 inside repaint region, 0.0 outside, linear ramps at edges."""
    mask = torch.zeros(total_samples, device=device, dtype=torch.float32)
    mask[start_sample:end_sample] = 1.0
    if crossfade_samples <= 0:
        return mask

    fade_start = max(start_sample - crossfade_samples, 0)
    ramp_len = start_sample - fade_start
    if ramp_len > 0:
        ramp = torch.linspace(0.0, 1.0, ramp_len + 2, device=device)[1:-1]
        mask[fade_start:start_sample] = ramp

    fade_end = min(end_sample + crossfade_samples, total_samples)
    ramp_len = fade_end - end_sample
    if ramp_len > 0:
        ramp = torch.linspace(1.0, 0.0, ramp_len + 2, device=device)[1:-1]
        mask[end_sample:fade_end] = ramp

    return mask


def _apply_repaint_waveform_splice_cpu(
    pred_wavs: torch.Tensor,
    src_wav: torch.Tensor,
    repainting_start: float,
    repainting_end: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
    crossfade_duration: float = 0.025,
) -> torch.Tensor:
    """Splice original pre-VAE user waveform outside ``[repainting_start, repainting_end]``.

    Aligned with ``acestep.core.generation.handler.repaint_waveform_splice`` to
    eliminate VAE reconstruction drift in unedited regions while smoothly
    crossfading repaint boundaries.
    """
    bsz = pred_wavs.shape[0]
    src = _normalize_waveform_comfy(src_wav)
    if src.shape[0] == 1 and bsz > 1:
        src = src.repeat(bsz, 1, 1)

    min_samples = min(pred_wavs.shape[-1], src.shape[-1])
    pred_trimmed = pred_wavs[..., :min_samples]
    src_trimmed = src[:bsz, :, :min_samples].to(device=pred_trimmed.device, dtype=pred_trimmed.dtype)

    start_sample = max(0, min(int(round(repainting_start * sample_rate)), min_samples))
    end_sample = (
        min_samples
        if repainting_end < 0
        else max(start_sample, min(int(round(repainting_end * sample_rate)), min_samples))
    )

    if start_sample == 0 and end_sample >= min_samples:
        return pred_wavs

    crossfade_samples = max(0, int(round(max(0.0, crossfade_duration) * sample_rate)))
    mask = _build_waveform_crossfade_mask(
        min_samples,
        start_sample,
        end_sample,
        crossfade_samples,
        device=pred_trimmed.device,
    )
    m = mask.unsqueeze(0).unsqueeze(0).expand_as(pred_trimmed)
    spliced = m * pred_trimmed + (1.0 - m) * src_trimmed

    if pred_wavs.shape[-1] > min_samples:
        spliced = torch.cat([spliced, pred_wavs[..., min_samples:]], dim=-1)
    return spliced


class AceStepVAEEncode:
    """Encode ComfyUI AUDIO to 25Hz ACE-Step 1.5 latents with strict VRAM isolation."""

    CATEGORY = "ACE-Step 1.5/VAE"
    RETURN_TYPES = ("ACESTEP_AUDIO_LATENT", "FLOAT")
    RETURN_NAMES = ("audio_latent", "duration_sec")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_model": (
                    "ACESTEP_VAE",
                    {"tooltip": "VAE handle from ACE-Step 1.5 VAE Loader."},
                ),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio from LoadAudio or any ComfyUI AUDIO node."},
                ),
                "use_tiled_encode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use overlap-discard tiled VAE encoding to prevent VRAM spikes "
                            "on long tracks (up to 600 s) on 12 GB GPUs."
                        ),
                    },
                ),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 4.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Audio tile length in seconds (15 s is safe and fast for 12 GB VRAM).",
                    },
                ),
                "overlap_seconds": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Overlap seconds on each side of a tile to eliminate boundary discontinuities.",
                    },
                ),
                "normalize_peak": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prevent clipping by scaling audio down if peak amplitude exceeds 1.0.",
                    },
                ),
                "fallback_to_cpu_on_oom": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically retry encoding on CPU float32 if CUDA OOM occurs.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Mandatory for 12 GB VRAM: unload VAE and purge CUDA/ComfyUI caches "
                            "immediately after encoding so DiT XL has clean VRAM."
                        ),
                    },
                ),
            },
            "optional": {
                "trim_start_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Optional start offset in seconds to crop from the input audio.",
                    },
                ),
                "trim_end_sec": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Optional end offset in seconds (-1 keeps full audio length).",
                    },
                ),
            },
        }

    def _encode_tiled(
        self,
        vae: Any,
        wav_bct: torch.Tensor,
        exec_device: torch.device,
        vae_dtype: torch.dtype,
        chunk_samples: int,
        overlap_samples: int,
    ) -> torch.Tensor:
        """Encode ``[1, 2, samples]`` into ``[1, 64, latent_frames]`` via overlap-discard tiling."""
        total_samples = wav_bct.shape[-1]
        eff_overlap = min(overlap_samples, max(0, (chunk_samples - LATENT_FRAME_HOP) // 2))
        core_step = max(LATENT_FRAME_HOP, chunk_samples - 2 * eff_overlap)

        if total_samples <= chunk_samples:
            with torch.inference_mode():
                x = wav_bct.to(device=exec_device, dtype=vae_dtype)
                latents = vae.encode(x).latent_dist.sample()
                return latents.detach().to(device="cpu", dtype=torch.float32)

        num_tiles = math.ceil(total_samples / core_step)
        pbar = ProgressBar(num_tiles) if ProgressBar is not None else None
        chunks: List[torch.Tensor] = []
        downsample_factor: float = float(LATENT_FRAME_HOP)
        calibrated = False

        core_start = 0
        with torch.inference_mode():
            while core_start < total_samples:
                core_end = min(total_samples, core_start + core_step)
                win_start = max(0, core_start - eff_overlap)
                win_end = min(total_samples, core_end + eff_overlap)

                x_chunk = wav_bct[:, :, win_start:win_end].to(device=exec_device, dtype=vae_dtype)
                z_chunk = vae.encode(x_chunk).latent_dist.sample()
                del x_chunk

                if not calibrated and z_chunk.shape[-1] > 0:
                    downsample_factor = (win_end - win_start) / float(z_chunk.shape[-1])
                    calibrated = True

                trim_left = int(round((core_start - win_start) / downsample_factor))
                trim_right = int(round((win_end - core_end) / downsample_factor))
                end_idx = z_chunk.shape[-1] - trim_right if trim_right > 0 else z_chunk.shape[-1]
                trimmed = z_chunk[..., trim_left:end_idx]
                chunks.append(trimmed.detach().to(device="cpu", dtype=torch.float32))
                del z_chunk, trimmed

                if exec_device.type == "cuda":
                    torch.cuda.empty_cache()
                if pbar is not None:
                    pbar.update(1)
                core_start = core_end

        return torch.cat(chunks, dim=-1)

    def encode(
        self,
        vae_model: AceStepVAEHandle,
        audio: Dict[str, Any],
        use_tiled_encode: bool,
        chunk_seconds: float,
        overlap_seconds: float,
        normalize_peak: bool,
        fallback_to_cpu_on_oom: bool,
        unload_after_run: bool,
        trim_start_sec: float = 0.0,
        trim_end_sec: float = -1.0,
    ) -> Tuple[Dict[str, Any], float]:
        wav_48k, duration_sec = _normalize_input_audio_stereo_48k(
            audio=audio,
            target_sr=TARGET_SAMPLE_RATE,
            normalize_peak=normalize_peak,
            trim_start_sec=trim_start_sec,
            trim_end_sec=trim_end_sec,
        )
        if wav_48k.shape[-1] < LATENT_FRAME_HOP:
            pad_len = LATENT_FRAME_HOP - wav_48k.shape[-1]
            wav_48k = F.pad(wav_48k, (0, pad_len))
            duration_sec = wav_48k.shape[-1] / float(TARGET_SAMPLE_RATE)

        chunk_samples = max(LATENT_FRAME_HOP * 4, int(round(chunk_seconds * TARGET_SAMPLE_RATE)))
        overlap_samples = max(0, int(round(overlap_seconds * TARGET_SAMPLE_RATE)))

        try:
            vae, exec_device, vae_dtype = vae_model.ensure_loaded()
            logger.info(
                "[ACE-Step 1.5] Encoding audio duration=%.2fs shape=%s on %s (%s, tiled=%s)...",
                duration_sec,
                tuple(wav_48k.shape),
                exec_device,
                vae_dtype,
                use_tiled_encode,
            )

            try:
                if use_tiled_encode:
                    latents_bct = self._encode_tiled(
                        vae=vae,
                        wav_bct=wav_48k,
                        exec_device=exec_device,
                        vae_dtype=vae_dtype,
                        chunk_samples=chunk_samples,
                        overlap_samples=overlap_samples,
                    )
                else:
                    with torch.inference_mode():
                        x = wav_48k.to(device=exec_device, dtype=vae_dtype)
                        latents_bct = vae.encode(x).latent_dist.sample().detach().to(device="cpu", dtype=torch.float32)
                        del x
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not (is_oom and fallback_to_cpu_on_oom and exec_device.type != "cpu"):
                    raise
                logger.warning(
                    "[ACE-Step 1.5] GPU OOM during VAE encode (%s); retrying cleanly on CPU float32...",
                    exc,
                )
                move_module_to_device_safe(vae, "cpu", torch.float32)
                purge_vram(aggressive=True)
                latents_bct = self._encode_tiled(
                    vae=vae,
                    wav_bct=wav_48k,
                    exec_device=torch.device("cpu"),
                    vae_dtype=torch.float32,
                    chunk_samples=chunk_samples,
                    overlap_samples=overlap_samples,
                )
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading VAE after encode (unload_after_run=True).")
                vae_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        # Store [1, frames_25hz, 64] on CPU alongside the resampled 48 kHz waveform
        latents_btc = latents_bct.transpose(1, 2).contiguous()
        payload: Dict[str, Any] = {
            "latents": latents_btc,             # [1, T_25hz, 64] float32 CPU
            "latents_bct": latents_bct.contiguous(),  # [1, 64, T_25hz] float32 CPU
            "waveform": wav_48k,                # [1, 2, samples] float32 CPU at 48 kHz
            "sample_rate": TARGET_SAMPLE_RATE,
            "latent_rate_hz": 25,
            "duration": float(duration_sec),
        }
        return (payload, float(duration_sec))


class AceStepVAEDecode:
    """Decode ACE-Step 1.5 25Hz latents to 48 kHz stereo ComfyUI AUDIO."""

    CATEGORY = "ACE-Step 1.5/VAE"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_model": (
                    "ACESTEP_VAE",
                    {"tooltip": "VAE handle from ACE-Step 1.5 VAE Loader."},
                ),
                "latent": (
                    "ACESTEP_LATENT",
                    {"tooltip": "25Hz latent payload produced by ACE-Step 1.5 DiT Sampler."},
                ),
                "use_tiled_decode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use overlap-discard tiled VAE decoding to keep VRAM usage low "
                            "even on 600-second tracks."
                        ),
                    },
                ),
                "chunk_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                        "tooltip": (
                            "Latent frames per decode tile (at 25 Hz, 256 frames ~= 10.2 seconds). "
                            "Lower values reduce peak VRAM further."
                        ),
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Overlap latent frames on each side of a tile to prevent boundary clicks.",
                    },
                ),
                "offload_wav_to_cpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Move each decoded waveform tile to CPU RAM immediately to prevent VRAM buildup.",
                    },
                ),
                "fallback_to_cpu_on_oom": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically retry decoding on CPU float32 if a CUDA Out-Of-Memory error occurs.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Unload VAE and purge CUDA/ComfyUI caches immediately after decoding "
                            "so VRAM returns cleanly to baseline."
                        ),
                    },
                ),
            },
            "optional": {
                "apply_repaint_splice": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For Repaint & Edit outputs, splice the bit-exact original waveform "
                            "outside the repainted window in CPU with linear boundary crossfades."
                        ),
                    },
                ),
                "repaint_crossfade_sec": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": (
                            "Override waveform splice crossfade duration in seconds. "
                            "-1 uses the automatic crossfade derived from the Repaint Task node."
                        ),
                    },
                ),
            },
        }

    def _decode_tiled(
        self,
        vae: Any,
        latents_bct: torch.Tensor,
        exec_device: torch.device,
        vae_dtype: torch.dtype,
        chunk_size: int,
        overlap: int,
        offload_wav_to_cpu: bool,
    ) -> torch.Tensor:
        """Execute overlap-discard tiled decoding over latent frames."""
        total_frames = latents_bct.shape[-1]

        # MPS Conv1d has a tight temporal output limit (aligned with acestep vae_decode.py)
        if exec_device.type == "mps":
            chunk_size = min(chunk_size, 32)
            overlap = min(overlap, 8)

        effective_overlap = min(overlap, max(0, (chunk_size - 1) // 2))
        core_step = max(1, chunk_size - 2 * effective_overlap)

        if total_frames <= chunk_size:
            with torch.inference_mode():
                z = latents_bct.to(device=exec_device, dtype=vae_dtype)
                out = vae.decode(z).sample
                return out.detach().cpu() if offload_wav_to_cpu else out.detach()

        num_tiles = math.ceil(total_frames / core_step)
        pbar = ProgressBar(num_tiles) if ProgressBar is not None else None

        decoded_chunks: List[torch.Tensor] = []
        upsample_factor: float = float(LATENT_FRAME_HOP)
        calibrated = False

        core_start = 0
        with torch.inference_mode():
            while core_start < total_frames:
                core_end = min(total_frames, core_start + core_step)
                win_start = max(0, core_start - effective_overlap)
                win_end = min(total_frames, core_end + effective_overlap)

                z_chunk = latents_bct[:, :, win_start:win_end].to(device=exec_device, dtype=vae_dtype)
                wav_chunk = vae.decode(z_chunk).sample
                del z_chunk

                if not calibrated:
                    win_frames = max(1, win_end - win_start)
                    upsample_factor = wav_chunk.shape[-1] / float(win_frames)
                    calibrated = True

                trim_left = int(round((core_start - win_start) * upsample_factor))
                trim_right = int(round((win_end - core_end) * upsample_factor))
                end_idx = wav_chunk.shape[-1] - trim_right if trim_right > 0 else wav_chunk.shape[-1]
                trimmed = wav_chunk[..., trim_left:end_idx]

                if offload_wav_to_cpu:
                    decoded_chunks.append(trimmed.detach().to(device="cpu", dtype=torch.float32))
                else:
                    decoded_chunks.append(trimmed.detach())

                del wav_chunk, trimmed
                if exec_device.type == "cuda":
                    torch.cuda.empty_cache()

                if pbar is not None:
                    pbar.update(1)
                core_start = core_end

        return torch.cat(decoded_chunks, dim=-1)

    def _decode_direct(
        self,
        vae: Any,
        latents_bct: torch.Tensor,
        exec_device: torch.device,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        with torch.inference_mode():
            z = latents_bct.to(device=exec_device, dtype=vae_dtype)
            out = vae.decode(z).sample
            del z
            return out.detach().cpu()

    def decode(
        self,
        vae_model: AceStepVAEHandle,
        latent: Dict[str, Any],
        use_tiled_decode: bool,
        chunk_size: int,
        overlap: int,
        offload_wav_to_cpu: bool,
        fallback_to_cpu_on_oom: bool,
        unload_after_run: bool,
        apply_repaint_splice: bool = True,
        repaint_crossfade_sec: float = -1.0,
    ) -> Tuple[Dict[str, Any]]:
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Invalid ACESTEP_LATENT input: expected a dictionary containing 'samples'.")

        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"Expected latent['samples'] to be a torch.Tensor, got {type(samples)!r}.")

        latents_bct = _normalize_latents_bct(samples)
        sample_rate = int(latent.get("sample_rate", vae_model.sample_rate or TARGET_SAMPLE_RATE))

        try:
            vae, exec_device, vae_dtype = vae_model.ensure_loaded()
            logger.info(
                "[ACE-Step 1.5] Decoding latents shape=%s on %s (%s, tiled=%s, chunk=%d, overlap=%d)...",
                tuple(latents_bct.shape),
                exec_device,
                vae_dtype,
                use_tiled_decode,
                chunk_size,
                overlap,
            )

            try:
                if use_tiled_decode:
                    raw_wav = self._decode_tiled(
                        vae=vae,
                        latents_bct=latents_bct,
                        exec_device=exec_device,
                        vae_dtype=vae_dtype,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        offload_wav_to_cpu=offload_wav_to_cpu,
                    )
                else:
                    raw_wav = self._decode_direct(
                        vae=vae,
                        latents_bct=latents_bct,
                        exec_device=exec_device,
                        vae_dtype=vae_dtype,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not (is_oom and fallback_to_cpu_on_oom and exec_device.type != "cpu"):
                    raise
                logger.warning(
                    "[ACE-Step 1.5] GPU OOM during VAE decode (%s); retrying cleanly on CPU float32...",
                    exc,
                )
                move_module_to_device_safe(vae, "cpu", torch.float32)
                purge_vram(aggressive=True)
                raw_wav = self._decode_tiled(
                    vae=vae,
                    latents_bct=latents_bct,
                    exec_device=torch.device("cpu"),
                    vae_dtype=torch.float32,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    offload_wav_to_cpu=True,
                )

            waveform = _normalize_waveform_comfy(raw_wav)
            del raw_wav

            # Optional CPU-side Repaint waveform splice (zero VRAM cost)
            splice_cfg = latent.get("repaint_splice")
            if (
                apply_repaint_splice
                and isinstance(splice_cfg, dict)
                and splice_cfg.get("enabled", False)
                and splice_cfg.get("src_waveform") is not None
            ):
                xfade = (
                    float(repaint_crossfade_sec)
                    if repaint_crossfade_sec >= 0.0
                    else float(splice_cfg.get("wav_crossfade_sec", 0.025))
                )
                logger.info(
                    "[ACE-Step 1.5] Applying Repaint waveform splice [%.2fs, %.2fs] (crossfade=%.3fs)...",
                    float(splice_cfg.get("repainting_start", 0.0)),
                    float(splice_cfg.get("repainting_end", -1.0)),
                    xfade,
                )
                waveform = _apply_repaint_waveform_splice_cpu(
                    pred_wavs=waveform,
                    src_wav=splice_cfg["src_waveform"],
                    repainting_start=float(splice_cfg.get("repainting_start", 0.0)),
                    repainting_end=float(splice_cfg.get("repainting_end", -1.0)),
                    sample_rate=sample_rate,
                    crossfade_duration=xfade,
                )
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading VAE after decode (unload_after_run=True).")
                vae_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        audio_output: Dict[str, Any] = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
        return (audio_output,)


NODE_CLASS_MAPPINGS = {
    "AceStepVAEEncode": AceStepVAEEncode,
    "AceStepVAEDecode": AceStepVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepVAEEncode": "ACE-Step 1.5 VAE Encode",
    "AceStepVAEDecode": "ACE-Step 1.5 VAE Decode",
}
