"""Memory management, checkpoint discovery, block offloading, and INT8 safetensors
quantization utilities for ComfyUI-Bao-Ace_Step_1_5.

All helpers in this module are designed to keep peak VRAM within 12 GB when
executing demanding ACE-Step 1.5 pipelines (e.g. 4B XL DiT + 4B 5Hz LM + VAE)
by strictly isolating each stage and aggressively releasing GPU memory through
``comfy.model_management``.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import comfy.model_management as comfy_mm
except ImportError:  # pragma: no cover - allows standalone syntax/unit testing outside ComfyUI
    comfy_mm = None

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")

# ==============================================================================
# Official ACE-Step 1.5 Checkpoint Registry (aligned with acestep.model_downloader)
# ==============================================================================

MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"

KNOWN_LM_MODELS: Dict[str, str] = {
    "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B": MAIN_MODEL_REPO,
    "acestep-5Hz-lm-4B": "ACE-Step/acestep-5Hz-lm-4B",
}

KNOWN_DIT_MODELS: Dict[str, str] = {
    "acestep-v15-turbo": MAIN_MODEL_REPO,
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-turbo-shift3": "ACE-Step/acestep-v15-turbo-shift3",
    "acestep-v15-turbo-shift1": "ACE-Step/acestep-v15-turbo-shift1",
    "acestep-v15-turbo-continuous": "ACE-Step/acestep-v15-turbo-continuous",
    "acestep-v15-xl-base": "ACE-Step/acestep-v15-xl-base",
    "acestep-v15-xl-sft": "ACE-Step/acestep-v15-xl-sft",
    "acestep-v15-xl-turbo": "ACE-Step/acestep-v15-xl-turbo",
}

KNOWN_VAE_MODELS: Dict[str, str] = {
    "vae": MAIN_MODEL_REPO,
    "scragvae": "scragnog/Ace-Step-1.5-ScragVAE",
}

TEXT_ENCODER_DIRNAME = "Qwen3-Embedding-0.6B"
QUANT_META_FILENAME = "quant_meta.json"
QUANT_WEIGHTS_FILENAME = "model_int8.safetensors"


# ==============================================================================
# Folder Paths & Checkpoint Discovery
# ==============================================================================


ACESTEP_FOLDER_KEYS: Tuple[str, ...] = ("acestep", "ace_step", "ace-step")
ACESTEP_FILE_EXTENSIONS = {".safetensors", ".pt", ".bin", ".json"}


def _norm_path_key(path: Path) -> str:
    """Return a normalized comparison key safe for non-existent paths on Linux/Windows/macOS."""
    return os.path.normcase(os.path.abspath(str(path.expanduser())))


def _get_default_comfy_acestep_dir() -> Optional[Path]:
    """Return ComfyUI's default ``<models_dir>/acestep`` path without creating it."""
    if folder_paths is None:
        return None
    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:
        return None
    return Path(models_dir) / "acestep"


def _collect_comfy_registered_acestep_paths() -> Tuple[List[Path], List[Path]]:
    """Inspect ``folder_paths`` for ``acestep`` paths configured via ``extra_model_paths.yaml``.

    Returns:
        Tuple ``(extra_paths, default_paths)`` preserving YAML registration order and
        ensuring ``folder_paths.folder_names_and_paths["acestep"]`` includes ACE-Step
        file extensions without appending ``ComfyUI/models/acestep`` when extra paths exist.
    """
    if folder_paths is None:
        return [], []

    default_dir = _get_default_comfy_acestep_dir()
    default_key = _norm_path_key(default_dir) if default_dir is not None else None

    fmap = getattr(folder_paths, "folder_names_and_paths", None)
    if not isinstance(fmap, dict):
        return [], []

    raw_paths: List[str] = []
    matched_keys: List[str] = []

    # Check canonical "acestep" first, then case-insensitive / common alias keys
    for key in ACESTEP_FOLDER_KEYS:
        if key in fmap:
            matched_keys.append(key)
    for key in list(fmap.keys()):
        if key not in matched_keys and key.lower() in ACESTEP_FOLDER_KEYS:
            matched_keys.append(key)

    for key in matched_keys:
        entry = fmap[key]
        if isinstance(entry, tuple) and len(entry) >= 1:
            paths_list = entry[0]
            exts = entry[1] if len(entry) >= 2 else None
            if isinstance(paths_list, (list, tuple)):
                raw_paths.extend(str(p) for p in paths_list if p)
            if isinstance(exts, set):
                exts.update(ACESTEP_FILE_EXTENSIONS)

    # Also consult folder_paths.get_folder_paths("acestep") if available
    if hasattr(folder_paths, "get_folder_paths"):
        for key in matched_keys or ["acestep"]:
            try:
                for p in folder_paths.get_folder_paths(key):
                    if p and str(p) not in raw_paths:
                        raw_paths.append(str(p))
            except KeyError:
                pass
            except Exception:
                pass

    extra_paths: List[Path] = []
    default_paths: List[Path] = []
    seen_keys: set[str] = set()

    for raw in raw_paths:
        p = Path(raw).expanduser()
        k = _norm_path_key(p)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        if default_key is not None and k == default_key:
            default_paths.append(p)
        else:
            extra_paths.append(p)

    return extra_paths, default_paths


def get_acestep_search_dirs() -> List[Path]:
    """Return ordered ACE-Step model root directories for reading/discovering checkpoints.

    Order of precedence:
    1. Paths configured for ``acestep`` in ComfyUI's ``extra_model_paths.yaml`` (in YAML order).
    2. ``ComfyUI/models/acestep`` ONLY if it already exists on disk (or as fallback when no
       extra paths are configured).
    3. ``ACESTEP_CHECKPOINTS_DIR`` environment variable / ``./models/acestep`` outside ComfyUI.
    """
    dirs: List[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None:
            return
        k = _norm_path_key(path)
        if k not in seen:
            seen.add(k)
            dirs.append(path)

    if folder_paths is not None:
        extra_paths, _default_paths = _collect_comfy_registered_acestep_paths()
        for p in extra_paths:
            _add(p)

        default_dir = _get_default_comfy_acestep_dir()
        if extra_paths:
            # Include ComfyUI/models/acestep for lookup ONLY if it already exists on disk;
            # never create an empty default directory when extra_model_paths.yaml is configured.
            if default_dir is not None and default_dir.is_dir():
                _add(default_dir)
        else:
            env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
            if env_dir:
                _add(Path(env_dir).expanduser().resolve())
            if default_dir is not None:
                _add(default_dir)
        return dirs

    env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env_dir:
        _add(Path(env_dir).expanduser().resolve())
    _add(Path.cwd() / "models" / "acestep")
    return dirs


def get_acestep_models_dir(create: bool = True) -> Path:
    """Return the primary ACE-Step models directory for downloads, writes, and INT8 exports.

    Strictly honors ``acestep`` paths configured in ComfyUI's ``extra_model_paths.yaml``
    (always prioritizing the configured extra path even without ``is_default: true``) and
    avoids creating an empty ``ComfyUI/models/acestep`` folder when an extra path exists.
    """
    if folder_paths is not None:
        extra_paths, _default_paths = _collect_comfy_registered_acestep_paths()
        if extra_paths:
            primary_dir = extra_paths[0]
            if create:
                primary_dir.mkdir(parents=True, exist_ok=True)
            # Ensure "acestep" key exists in folder_paths with our extensions (pointing to extra_paths)
            if "acestep" not in folder_paths.folder_names_and_paths:
                folder_paths.folder_names_and_paths["acestep"] = (
                    [str(p) for p in extra_paths],
                    set(ACESTEP_FILE_EXTENSIONS),
                )
            return primary_dir

        env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
        if env_dir:
            primary_dir = Path(env_dir).expanduser().resolve()
        else:
            primary_dir = _get_default_comfy_acestep_dir() or (Path.cwd() / "models" / "acestep")

        if create:
            primary_dir.mkdir(parents=True, exist_ok=True)

        if "acestep" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["acestep"] = (
                [str(primary_dir)],
                set(ACESTEP_FILE_EXTENSIONS),
            )
        else:
            existing_paths, exts = folder_paths.folder_names_and_paths["acestep"]
            if isinstance(exts, set):
                exts.update(ACESTEP_FILE_EXTENSIONS)
            primary_key = _norm_path_key(primary_dir)
            if not any(_norm_path_key(Path(p)) == primary_key for p in existing_paths):
                existing_paths.append(str(primary_dir))
        return primary_dir

    env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    fallback = Path.cwd() / "models" / "acestep"
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _dir_has_checkpoint_files(path: Path) -> bool:
    """Return True if a directory looks like a valid model or quantized checkpoint."""
    if not path.is_dir():
        return False
    expected_markers = (
        "config.json",
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        QUANT_META_FILENAME,
        QUANT_WEIGHTS_FILENAME,
    )
    for marker in expected_markers:
        if (path / marker).exists():
            return True
    return any(path.glob("*.safetensors"))


def list_checkpoint_choices(category: str) -> List[str]:
    """List local checkpoints across all configured ACE-Step directories merged with official presets.

    Args:
        category: One of ``"lm"``, ``"dit"``, or ``"vae"``.
    """
    local_matches: List[str] = []

    for models_dir in get_acestep_search_dirs():
        if not models_dir.is_dir():
            continue
        for entry in sorted(models_dir.iterdir()):
            name = entry.name
            lower = name.lower()
            if entry.is_dir() and _dir_has_checkpoint_files(entry):
                if category == "lm" and ("5hz-lm" in lower or "lm-" in lower or lower.startswith("lm")):
                    local_matches.append(name)
                elif category == "dit" and (
                    "acestep-v15" in lower
                    or "dit" in lower
                    or (entry / "silence_latent.pt").exists()
                ):
                    local_matches.append(name)
                elif category == "vae" and (
                    "vae" in lower or "oobleck" in lower or "scragvae" in lower
                ):
                    local_matches.append(name)
            elif entry.is_file() and entry.suffix == ".safetensors":
                if category == "dit" and ("acestep" in lower or "dit" in lower):
                    local_matches.append(name)
                elif category == "vae" and "vae" in lower:
                    local_matches.append(name)

    if category == "lm":
        official = list(KNOWN_LM_MODELS.keys())
    elif category == "dit":
        official = list(KNOWN_DIT_MODELS.keys())
    elif category == "vae":
        official = list(KNOWN_VAE_MODELS.keys())
    else:
        official = []

    merged: List[str] = []
    for item in local_matches + official:
        if item not in merged:
            merged.append(item)
    return merged


def resolve_checkpoint_path(model_path: str) -> Path:
    """Resolve a checkpoint across all configured ACE-Step directories, or default to primary dir."""
    candidate = Path(model_path).expanduser()
    if candidate.is_absolute():
        return candidate

    for root in get_acestep_search_dirs():
        found = root / model_path
        if found.is_file() or (found.is_dir() and _dir_has_checkpoint_files(found)):
            return found

    return get_acestep_models_dir(create=False) / model_path


def _ensure_acestep_package() -> None:
    """Verify the official ``acestep`` package is importable."""
    try:
        import acestep  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The official 'acestep' package is not installed in ComfyUI's Python environment. "
            "Install it with: pip install -r ComfyUI/custom_nodes/ComfyUI-Bao-Ace_Step_1_5/requirements.txt"
        ) from exc


def ensure_component_available(
    model_name: str,
    category: str,
    auto_download: bool = True,
) -> Path:
    """Ensure a requested model component exists locally across configured ACE-Step directories.

    If ``auto_download`` is True and the checkpoint is missing from all configured paths,
    downloads directly into ``get_acestep_models_dir()`` (honoring ``extra_model_paths.yaml``).
    """
    existing = resolve_checkpoint_path(model_name)
    if existing.is_file() or (existing.is_dir() and _dir_has_checkpoint_files(existing)):
        return existing

    models_dir = get_acestep_models_dir(create=auto_download)
    candidate_path = Path(model_name).expanduser()
    resolved = candidate_path if candidate_path.is_absolute() else (models_dir / model_name)

    if not auto_download:
        searched = ", ".join(str(p) for p in get_acestep_search_dirs())
        raise FileNotFoundError(
            f"Checkpoint '{model_name}' was not found in [{searched}] and auto_download is disabled. "
            f"Place the checkpoint in '{models_dir}' or enable auto_download."
        )

    logger.info(
        "[ACE-Step 1.5] Checkpoint '%s' missing locally; downloading into '%s'...",
        model_name,
        models_dir,
    )

    # For components hosted inside the shared MAIN_MODEL_REPO (e.g. Qwen3-Embedding-0.6B,
    # vae, acestep-v15-turbo, acestep-5Hz-lm-1.7B), download ONLY the requested subfolder
    # via Hugging Face allow_patterns so we never re-download sibling models the user deleted.
    if category == "text_encoder":
        repo_id: Optional[str] = MAIN_MODEL_REPO
    else:
        repo_id = (
            KNOWN_LM_MODELS.get(model_name)
            or KNOWN_DIT_MODELS.get(model_name)
            or KNOWN_VAE_MODELS.get(model_name)
        )
        if repo_id is None and "/" in model_name:
            repo_id = model_name

    msg = ""
    if repo_id == MAIN_MODEL_REPO:
        from huggingface_hub import snapshot_download

        if category == "text_encoder":
            subdir = TEXT_ENCODER_DIRNAME
        elif model_name in ("vae", "official"):
            subdir = "vae"
        else:
            subdir = model_name

        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[ACE-Step 1.5] Downloading isolated subfolder '%s/*' from '%s' into '%s'...",
            subdir,
            MAIN_MODEL_REPO,
            models_dir,
        )
        snapshot_download(
            repo_id=MAIN_MODEL_REPO,
            local_dir=str(models_dir),
            allow_patterns=[f"{subdir}/*"],
        )
        if resolved.is_file() or _dir_has_checkpoint_files(resolved):
            return resolved
        msg = f"Subfolder '{subdir}/*' downloaded from {MAIN_MODEL_REPO}, but checkpoint files were not found at '{resolved}'."

    _ensure_acestep_package()
    from acestep import model_downloader

    if category == "lm":
        ok, msg = model_downloader.ensure_lm_model(model_name, checkpoints_dir=models_dir)
    elif category == "dit":
        ok, msg = model_downloader.ensure_dit_model(model_name, checkpoints_dir=models_dir)
    elif category == "vae":
        vae_variant = "official" if model_name in ("vae", "official") else model_name
        ok, msg = model_downloader.ensure_vae_model(vae_variant, checkpoints_dir=models_dir)
    elif category == "text_encoder":
        ok = False
    else:
        ok, msg = False, f"Unknown category: {category}"

    if ok and (resolved.is_file() or _dir_has_checkpoint_files(resolved)):
        return resolved

    # Fallback direct HuggingFace snapshot download for standalone repos or explicit repo IDs
    if repo_id is not None and repo_id != MAIN_MODEL_REPO:
        from huggingface_hub import snapshot_download

        resolved.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(resolved),
        )
        if resolved.is_file() or _dir_has_checkpoint_files(resolved):
            return resolved

    raise RuntimeError(
        f"Failed to download ACE-Step 1.5 {category} checkpoint '{model_name}': {msg}"
    )


def ensure_text_encoder_available(auto_download: bool = True) -> Path:
    """Ensure ``Qwen3-Embedding-0.6B`` exists inside ``models/acestep/`` without pulling extra models."""
    return ensure_component_available(
        TEXT_ENCODER_DIRNAME,
        category="text_encoder",
        auto_download=auto_download,
    )


# ==============================================================================
# Strict ComfyUI VRAM Management & Safe Device Transfer
# ==============================================================================


def get_compute_device(device_pref: str = "cuda") -> torch.device:
    """Resolve the active PyTorch compute device."""
    if device_pref == "cpu":
        return torch.device("cpu")
    if comfy_mm is not None:
        try:
            dev = comfy_mm.get_torch_device()
            if device_pref in ("cuda", "auto", "auto_offload"):
                return dev
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Map precision string to a device-safe floating-point compute dtype."""
    if device.type == "cpu":
        return torch.float32 if precision == "fp16" else torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    # bf16 or int8_dynamic compute activations
    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(device)
            if major < 8:
                logger.warning(
                    "[ACE-Step 1.5] Pre-Ampere CUDA GPU detected (sm_%d); falling back to float16 activations.",
                    major * 10,
                )
                return torch.float16
        except Exception:
            pass
    return torch.bfloat16


def get_free_vram_gb(device: Optional[torch.device] = None) -> float:
    """Return currently free GPU VRAM in gigabytes (or infinity on CPU)."""
    if not torch.cuda.is_available():
        return float("inf")
    try:
        idx = device.index if (device is not None and device.index is not None) else torch.cuda.current_device()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
        return free_bytes / (1024.0 ** 3)
    except Exception:
        return float("inf")


def request_vram_headroom(needed_gb: float, device: Optional[torch.device] = None) -> None:
    """Ask ComfyUI to evict cached models from other nodes before loading an ACE-Step stage."""
    target_device = device or get_compute_device("cuda")
    if target_device.type != "cuda":
        return
    if comfy_mm is not None:
        try:
            free_gb = get_free_vram_gb(target_device)
            if free_gb < needed_gb:
                comfy_mm.free_memory(int(needed_gb * (1024 ** 3)), target_device)
                comfy_mm.soft_empty_cache()
        except Exception as exc:
            logger.debug("[ACE-Step 1.5] comfy.model_management.free_memory skipped: %s", exc)
    purge_vram(aggressive=False)


def purge_vram(aggressive: bool = True) -> None:
    """Execute the mandatory VRAM cleanup sequence across Python, PyTorch, and ComfyUI."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if aggressive and hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if comfy_mm is not None:
        try:
            comfy_mm.soft_empty_cache()
        except Exception:
            pass
    if aggressive:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_quantized_subclass(tensor: torch.Tensor) -> bool:
    """Detect torchao AffineQuantizedTensor or other non-standard Tensor subclasses."""
    return type(tensor) is not torch.Tensor and type(tensor) is not nn.Parameter


_FP32_PRECISION_BUFFER_NAMES = frozenset(
    {
        "weight_scale",
        "inv_freq",
        "implicit_codebook",
        "scales",
        "orthogonal_rot",
    }
)


def _is_fsq_module(mod: nn.Module) -> bool:
    """Return True if ``mod`` is a ResidualFSQ / GroupedResidualFSQ / FSQ module."""
    cls_name = mod.__class__.__name__.lower()
    return (
        "fsq" in cls_name
        or (hasattr(mod, "get_output_from_indices") and hasattr(mod, "get_codes_from_indices"))
        or (hasattr(mod, "_levels") and hasattr(mod, "_basis"))
    )


def _is_precision_sensitive_fp32_buffer(mod: nn.Module, buffer_name: str) -> bool:
    """Return True for floating-point buffers that must remain FP32 for numerical integrity.

    Preserves:
    - ``Int8DynamicLinear.weight_scale`` (per-channel dequantization scales)
    - RoPE ``inv_freq`` buffers (rotary positional embedding frequencies)
    - ResidualFSQ / FSQ codebook & scale buffers (``implicit_codebook``, ``scales``, ``orthogonal_rot``)
    """
    if buffer_name in _FP32_PRECISION_BUFFER_NAMES:
        return True
    cls_name = mod.__class__.__name__.lower()
    if "rotary" in cls_name and "freq" in buffer_name:
        return True
    if _is_fsq_module(mod) and buffer_name in ("codebooks", "implicit_codebook", "scales", "orthogonal_rot"):
        return True
    return False


def _enforce_fsq_fp32_buffers(quantizer: nn.Module) -> None:
    """Keep precision-sensitive floating buffers inside ``ResidualFSQ``/``FSQ`` in FP32."""
    for sub in quantizer.modules():
        for bname, buf in list(sub._buffers.items()):
            if buf is None or not buf.is_floating_point():
                continue
            if _is_precision_sensitive_fp32_buffer(sub, bname) and buf.dtype != torch.float32:
                sub._buffers[bname] = buf.to(dtype=torch.float32)


def _install_linear_input_dtype_hook(mod: Optional[nn.Module]) -> None:
    """Attach an idempotent pre-hook casting floating inputs to match a projection's weight dtype."""
    if mod is None or isinstance(mod, nn.Identity):
        return
    if getattr(mod, "_acestep_dtype_prehook_installed", False):
        return
    weight = getattr(mod, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        return

    def _align_input_pre_hook(m: nn.Module, args: Tuple[Any, ...]):
        if not args:
            return args
        x = args[0]
        w = getattr(m, "weight", None)
        if (
            isinstance(x, torch.Tensor)
            and x.is_floating_point()
            and isinstance(w, torch.Tensor)
            and w.is_floating_point()
            and x.dtype != w.dtype
        ):
            return (x.to(dtype=w.dtype),) + args[1:]
        return args

    mod.register_forward_pre_hook(_align_input_pre_hook)
    setattr(mod, "_acestep_dtype_prehook_installed", True)


def install_fsq_dtype_guard(
    model: Optional[nn.Module],
    compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """Install idempotent dtype guards on ResidualFSQ/FSQ quantizers inside an ACE-Step DiT model.

    Keeps FSQ lookup buffers (``implicit_codebook``, ``scales``) in FP32 for exact codebook math
    while automatically aligning tensor dtypes at the ``project_in``, ``project_out``,
    ``get_output_from_indices``, and ``forward`` boundaries. This eliminates
    ``RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16``
    across all 2B and XL DiT models in ``bf16``, ``fp16``, and ``int8_convrot`` modes.
    """
    if model is None:
        return

    quantizers: List[nn.Module] = []
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "quantizer", None) is not None:
        quantizers.append(tokenizer.quantizer)

    for sub in model.modules():
        if _is_fsq_module(sub) and sub not in quantizers:
            quantizers.append(sub)

    for quantizer in quantizers:
        if compute_dtype is not None:
            setattr(quantizer, "_acestep_target_dtype", compute_dtype)

        # Keep FSQ codebook/scale buffers in FP32
        _enforce_fsq_fp32_buffers(quantizer)

        # Protect every project_in / project_out linear layer inside ResidualFSQ & child FSQ layers
        for sub in quantizer.modules():
            _install_linear_input_dtype_hook(getattr(sub, "project_in", None))
            _install_linear_input_dtype_hook(getattr(sub, "project_out", None))

        if getattr(quantizer, "_acestep_fsq_dtype_guarded", False):
            continue

        # Wrap get_output_from_indices (called by _decode_audio_codes_to_latents)
        if hasattr(quantizer, "get_output_from_indices") and callable(quantizer.get_output_from_indices):
            orig_get_output = quantizer.get_output_from_indices

            def _make_guarded_get_output(q_mod: nn.Module, orig_fn: Callable[..., Any]):
                def _guarded_get_output_from_indices(indices: torch.Tensor, *args: Any, **kwargs: Any):
                    _enforce_fsq_fp32_buffers(q_mod)
                    proj_out = getattr(q_mod, "project_out", None)
                    proj_w = getattr(proj_out, "weight", None)
                    target_dev = proj_w.device if isinstance(proj_w, torch.Tensor) else None
                    if target_dev is None:
                        for p in q_mod.parameters():
                            target_dev = p.device
                            break
                    if target_dev is None:
                        for b in q_mod.buffers():
                            target_dev = b.device
                            break
                    if isinstance(indices, torch.Tensor) and target_dev is not None and indices.device != target_dev:
                        indices = indices.to(device=target_dev)

                    out = orig_fn(indices, *args, **kwargs)
                    desired_dtype = (
                        proj_w.dtype
                        if isinstance(proj_w, torch.Tensor) and proj_w.is_floating_point()
                        else getattr(q_mod, "_acestep_target_dtype", None)
                    )
                    if (
                        desired_dtype is not None
                        and isinstance(out, torch.Tensor)
                        and out.is_floating_point()
                        and out.dtype != desired_dtype
                    ):
                        out = out.to(dtype=desired_dtype)
                    return out

                return _guarded_get_output_from_indices

            quantizer.get_output_from_indices = _make_guarded_get_output(quantizer, orig_get_output)  # type: ignore[method-assign]

        # Wrap forward() so 2B models (which lack XL's fp32 cast) and XL models (which call
        # self.quantizer.to(quantizer_dtype) afterwards) never clash or degrade FSQ buffers.
        orig_forward = getattr(quantizer, "forward", None)
        if callable(orig_forward):
            def _make_guarded_forward(q_mod: nn.Module, orig_fn: Callable[..., Any]):
                def _guarded_forward(x: torch.Tensor, *args: Any, **kwargs: Any):
                    _enforce_fsq_fp32_buffers(q_mod)
                    in_dtype = x.dtype if isinstance(x, torch.Tensor) and x.is_floating_point() else None
                    res = orig_fn(x, *args, **kwargs)
                    _enforce_fsq_fp32_buffers(q_mod)
                    if in_dtype is not None:
                        if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], torch.Tensor):
                            if res[0].is_floating_point() and res[0].dtype != in_dtype:
                                res = (res[0].to(dtype=in_dtype),) + res[1:]
                        elif isinstance(res, torch.Tensor) and res.is_floating_point() and res.dtype != in_dtype:
                            res = res.to(dtype=in_dtype)
                    return res

                return _guarded_forward

            quantizer.forward = _make_guarded_forward(quantizer, orig_forward)  # type: ignore[method-assign]
        setattr(quantizer, "_acestep_fsq_dtype_guarded", True)

    # Also restore FP32 FSQ buffers after AceStepAudioTokenizer.forward() in XL models
    # (upstream XL calls `self.quantizer.to(quantizer_dtype)` at the end of forward).
    if tokenizer is not None and not getattr(tokenizer, "_acestep_tok_posthook_installed", False):
        def _tokenizer_post_hook(tok_mod: nn.Module, _args: Any, output: Any):
            q = getattr(tok_mod, "quantizer", None)
            if q is not None:
                _enforce_fsq_fp32_buffers(q)
            return output

        tokenizer.register_forward_hook(_tokenizer_post_hook)
        setattr(tokenizer, "_acestep_tok_posthook_installed", True)


def align_module_dtypes(
    module: Optional[nn.Module],
    compute_dtype: torch.dtype,
) -> None:
    """Coherently cast all non-quantized floating parameters and ordinary buffers to ``compute_dtype``
    in-place (preserving each tensor's current device), while keeping precision-sensitive buffers
    (``weight_scale``, RoPE ``inv_freq``, ResidualFSQ ``implicit_codebook``/``scales``) in FP32.
    """
    if module is None:
        return

    visited: set[int] = set()

    def _recurse(mod: nn.Module) -> None:
        mod_id = id(mod)
        if mod_id in visited:
            return
        visited.add(mod_id)

        if isinstance(mod, Int8DynamicLinear):
            mod.compute_dtype = compute_dtype

        for pname, param in list(mod._parameters.items()):
            if param is None or param.device.type == "meta":
                continue
            if _is_quantized_subclass(param.data):
                continue
            if param.is_floating_point() and param.dtype != compute_dtype:
                param.data = param.data.to(dtype=compute_dtype)

        for bname, buf in list(mod._buffers.items()):
            if buf is None or buf.device.type == "meta" or not buf.is_floating_point():
                continue
            target_buf_dtype = (
                torch.float32 if _is_precision_sensitive_fp32_buffer(mod, bname) else compute_dtype
            )
            if buf.dtype != target_buf_dtype:
                mod._buffers[bname] = buf.to(dtype=target_buf_dtype)

        for child in mod._modules.values():
            if child is not None:
                _recurse(child)

    _recurse(module)
    install_fsq_dtype_guard(module, compute_dtype=compute_dtype)


def move_module_to_device_safe(
    module: Optional[nn.Module],
    target_device: str | torch.device,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Safely transfer a module (including torchao quantized tensors and Int8DynamicLinear)
    between CUDA and CPU with coherent dtype casting while preserving INT8 weights and
    precision-sensitive FP32 buffers (``weight_scale``, RoPE ``inv_freq``, FSQ codebooks/scales).
    """
    if module is None:
        return

    dev = torch.device(target_device) if isinstance(target_device, str) else target_device

    # Fast path first (moves device without corrupting INT8 or FP32 precision-sensitive buffers)
    fast_failed = False
    try:
        module.to(device=dev)
    except NotImplementedError:
        fast_failed = True

    visited: set[int] = set()

    def _recurse(mod: nn.Module) -> None:
        mod_id = id(mod)
        if mod_id in visited:
            return
        visited.add(mod_id)

        if dtype is not None and isinstance(mod, Int8DynamicLinear):
            mod.compute_dtype = dtype

        for pname, param in list(mod._parameters.items()):
            if param is None:
                continue
            if _is_quantized_subclass(param.data):
                if param.device != dev:
                    if hasattr(param.data, "_apply_fn_to_data"):
                        moved_data = param.data._apply_fn_to_data(lambda x: x.to(dev))
                    else:
                        moved_data = param.data.to(dev)
                    mod._parameters[pname] = nn.Parameter(moved_data, requires_grad=param.requires_grad)
            else:
                needs_move = param.device != dev
                needs_cast = (
                    dtype is not None
                    and param.is_floating_point()
                    and param.dtype != dtype
                )
                if needs_move or needs_cast:
                    new_data = param.data.to(device=dev, dtype=dtype if needs_cast else param.dtype)
                    param.data = new_data

        for bname, buf in list(mod._buffers.items()):
            if buf is None:
                continue
            if dtype is not None and buf.is_floating_point():
                target_buf_dtype = (
                    torch.float32 if _is_precision_sensitive_fp32_buffer(mod, bname) else dtype
                )
            else:
                target_buf_dtype = buf.dtype
            if buf.device != dev or buf.dtype != target_buf_dtype:
                mod._buffers[bname] = buf.to(device=dev, dtype=target_buf_dtype)

        for child in mod._modules.values():
            if child is not None:
                _recurse(child)

    if fast_failed or dtype is not None:
        _recurse(module)
    else:
        # Verification sweep for quantized tensor subclasses or buffers that .to() might leave behind
        needs_sweep = False
        for _, p in module.named_parameters():
            if p.device != dev:
                needs_sweep = True
                break
        if not needs_sweep:
            for _, b in module.named_buffers():
                if b is not None and b.device != dev:
                    needs_sweep = True
                    break
        if needs_sweep:
            _recurse(module)

    if dtype is not None:
        install_fsq_dtype_guard(module, compute_dtype=dtype)


def install_handler_safe_transfer(handler: Any) -> None:
    """Bind ``handler._recursive_to_device`` to ``move_module_to_device_safe``.

    Ensures both 12 GB (block offload) and >= 24 GB (full resident / handler offload) GPUs
    never corrupt ``Int8DynamicLinear.weight_scale``, RoPE ``inv_freq``, or ResidualFSQ
    FP32 buffers when upstream ``_load_model_context`` moves models between CPU and CUDA.
    """
    if handler is None or getattr(handler, "_acestep_safe_transfer_installed", False):
        return

    def _safe_recursive_to_device(model: Any, device: Any, dtype: Any = None) -> None:
        move_module_to_device_safe(model, device, dtype)

    handler._recursive_to_device = _safe_recursive_to_device
    setattr(handler, "_acestep_safe_transfer_installed", True)


# ==============================================================================
# Dynamic Block-Level Offloading for DiT (12 GB VRAM Safeguard)
# ==============================================================================


class DiTBlockOffloadManager:
    """Layer/block-level CPU offload hook manager for ACE-Step 1.5 DiT models.

    When ``offload_dit_to_cpu`` is enabled and free VRAM drops below
    ``vram_threshold_gb``, transformer blocks inside the DiT decoder are kept on
    CPU and streamed to GPU strictly for the duration of their individual
    ``forward`` call, preventing OOM on 12 GB GPUs even with 4B XL models and
    long durations (up to 600 s).
    """

    def __init__(
        self,
        model: nn.Module,
        exec_device: torch.device,
        compute_dtype: torch.dtype,
        vram_threshold_gb: float = 1.5,
        force_block_offload: bool = False,
    ) -> None:
        self.model = model
        self.exec_device = exec_device
        self.compute_dtype = compute_dtype
        self.vram_threshold_gb = max(0.25, float(vram_threshold_gb))
        self.force_block_offload = force_block_offload
        self._handles: List[Any] = []
        self._offloaded_blocks: List[nn.Module] = []
        self._active = False

    def _discover_transformer_blocks(self) -> List[nn.Module]:
        """Locate the main sequential transformer block list inside the DiT decoder."""
        decoder = getattr(self.model, "decoder", self.model)
        candidate_attrs = ("layers", "blocks", "transformer_blocks", "dit_blocks")
        for attr in candidate_attrs:
            sub = getattr(decoder, attr, None)
            if isinstance(sub, nn.ModuleList) and len(sub) > 0:
                return list(sub)

        # Fallback: find the largest ModuleList inside decoder
        best: List[nn.Module] = []
        for _, sub in decoder.named_modules():
            if isinstance(sub, nn.ModuleList) and len(sub) > len(best):
                best = list(sub)
        return best

    def _move_inputs_to_device(self, obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True) if obj.device != device else obj
        if isinstance(obj, tuple):
            return tuple(self._move_inputs_to_device(x, device) for x in obj)
        if isinstance(obj, list):
            return [self._move_inputs_to_device(x, device) for x in obj]
        if isinstance(obj, dict):
            return {k: self._move_inputs_to_device(v, device) for k, v in obj.items()}
        return obj

    def attach_if_needed(self) -> bool:
        """Activate block-level offloading if requested or if free VRAM is below threshold."""
        if self.exec_device.type != "cuda":
            return False

        free_gb = get_free_vram_gb(self.exec_device)
        blocks = self._discover_transformer_blocks()
        if not blocks:
            logger.debug("[ACE-Step 1.5] No sequential transformer blocks found for block offload.")
            return False

        if not self.force_block_offload and free_gb >= self.vram_threshold_gb:
            logger.info(
                "[ACE-Step 1.5] Free VRAM %.2f GB >= threshold %.2f GB; keeping DiT resident on GPU.",
                free_gb,
                self.vram_threshold_gb,
            )
            return False

        logger.info(
            "[ACE-Step 1.5] Activating DiT block-level CPU offload (%d blocks, free VRAM %.2f GB, threshold %.2f GB).",
            len(blocks),
            free_gb,
            self.vram_threshold_gb,
        )

        # Keep lightweight non-block layers (embeddings, projections, norms, tokenizer) on GPU
        # and move heavy transformer blocks to CPU.
        move_module_to_device_safe(self.model, self.exec_device, self.compute_dtype)
        for blk in blocks:
            move_module_to_device_safe(blk, "cpu")
            self._offloaded_blocks.append(blk)

            def _pre_hook(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                move_module_to_device_safe(mod, self.exec_device)
                return (
                    self._move_inputs_to_device(args, self.exec_device),
                    self._move_inputs_to_device(kwargs, self.exec_device),
                )

            def _post_hook(mod: nn.Module, _args: Tuple[Any, ...], _kwargs: Dict[str, Any], output: Any):
                move_module_to_device_safe(mod, "cpu")
                if get_free_vram_gb(self.exec_device) < self.vram_threshold_gb * 0.5:
                    torch.cuda.empty_cache()
                return output

            self._handles.append(blk.register_forward_pre_hook(_pre_hook, with_kwargs=True))
            self._handles.append(blk.register_forward_hook(_post_hook, with_kwargs=True))

        purge_vram(aggressive=False)
        self._active = True
        return True

    def detach(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._offloaded_blocks.clear()
        self._active = False


# ==============================================================================
# INT8 Dynamic Quantization & Safetensors Persistence Engine
# ==============================================================================


class Int8DynamicLinear(nn.Module):
    """Per-output-channel symmetric INT8 linear layer with on-the-fly dequantization.

    Stores weights natively as ``torch.int8`` + ``float32`` per-channel scale
    buffers so they serialize cleanly into standard ``.safetensors`` files
    without pickle, work identically on Python 3.12 and 3.13, and support
    instantaneous CPU<->CUDA transfers during strict 12 GB VRAM offloading.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=compute_dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "Int8DynamicLinear":
        """Quantize a floating-point ``nn.Linear`` layer into an ``Int8DynamicLinear``."""
        with torch.no_grad():
            w = linear.weight.detach().to(torch.float32)
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / 127.0
            q_weight = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)

            layer = cls(
                in_features=linear.in_features,
                out_features=linear.out_features,
                has_bias=linear.bias is not None,
                compute_dtype=compute_dtype,
            )
            layer.weight_int8.copy_(q_weight.to(linear.weight.device))
            layer.weight_scale.copy_(scale.to(linear.weight.device))
            if linear.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(linear.bias.detach().to(device=linear.weight.device, dtype=compute_dtype))
            return layer

    @classmethod
    def from_quantized_tensors(
        cls,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "Int8DynamicLinear":
        out_features, in_features = weight_int8.shape
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            has_bias=bias is not None,
            compute_dtype=compute_dtype,
        )
        layer.weight_int8 = weight_int8.to(torch.int8).contiguous()
        scale = weight_scale.to(torch.float32)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        layer.weight_scale = scale.contiguous()
        if bias is not None:
            layer.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype if x.is_floating_point() else self.compute_dtype
        # Per-channel dequantization into activation dtype; memory overhead is transient per layer.
        weight_fp = self.weight_int8.to(dtype=target_dtype) * self.weight_scale.to(dtype=target_dtype)
        bias = self.bias.to(dtype=target_dtype) if self.bias is not None else None
        return F.linear(x, weight_fp, bias)


def build_quant_filter_fn(scope: str) -> Callable[[nn.Module, str], bool]:
    """Return a module filter matching upstream ACE-Step 1.5 quantization policy.

    - For ``dit``: quantizes only ``decoder.*`` ``nn.Linear`` layers while
      strictly excluding ``tokenizer`` and ``detokenizer`` (preserving FSQ audio fidelity).
    - For ``lm``: quantizes transformer ``nn.Linear`` layers while excluding
      ``lm_head`` and embedding layers.
    """
    def _filter(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        parts = fqn.split(".")
        if scope == "dit":
            if not parts or parts[0] != "decoder":
                return False
            for part in parts:
                if part in ("tokenizer", "detokenizer"):
                    return False
            return True
        if scope == "lm":
            for part in parts:
                if part in ("lm_head", "embed_tokens", "wte", "wpe"):
                    return False
            return True
        return True

    return _filter


def _try_torchao_quantize(model: nn.Module, scope: str) -> Tuple[bool, str]:
    """Attempt native ``torchao`` INT8 dynamic quantization (Python 3.12 / 3.13 preferred path)."""
    try:
        from torchao.quantization import quantize_
    except Exception as exc:
        logger.warning("[ACE-Step 1.5] torchao unavailable (%s); using Int8DynamicLinear fallback.", exc)
        return False, "int8_linear_fallback"

    quant_config = None
    backend_tag = "torchao_int8_dynamic"

    # Prefer Int8DynamicActivationInt8WeightConfig (matches ACE-Step w8a8_dynamic),
    # then Int8WeightOnlyConfig depending on torchao version.
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
        quant_config = Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
        backend_tag = "torchao_int8_dynamic_a8w8"
    except Exception:
        try:
            from torchao.quantization import Int8WeightOnlyConfig
            quant_config = Int8WeightOnlyConfig()
            backend_tag = "torchao_int8_weight_only"
        except Exception:
            try:
                from torchao.quantization import int8_dynamic_activation_int8_weight
                quant_config = int8_dynamic_activation_int8_weight()
                backend_tag = "torchao_int8_dynamic_legacy"
            except Exception as exc:
                logger.warning(
                    "[ACE-Step 1.5] No compatible torchao INT8 config found (%s); using Int8DynamicLinear fallback.",
                    exc,
                )
                return False, "int8_linear_fallback"

    filter_fn = build_quant_filter_fn(scope)
    try:
        quantize_(model, quant_config, filter_fn=filter_fn)
        logger.info("[ACE-Step 1.5] Applied %s quantization to %s model.", backend_tag, scope)
        return True, backend_tag
    except Exception as exc:
        logger.warning(
            "[ACE-Step 1.5] torchao quantize_() failed (%s); falling back to Int8DynamicLinear.",
            exc,
        )
        return False, "int8_linear_fallback"


def _set_submodule(root: nn.Module, fqn: str, new_module: nn.Module) -> None:
    parts = fqn.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _apply_int8_linear_fallback(
    model: nn.Module,
    scope: str,
    compute_dtype: torch.dtype,
) -> int:
    """Replace filtered ``nn.Linear`` modules with ``Int8DynamicLinear`` in-place."""
    filter_fn = build_quant_filter_fn(scope)
    targets: List[Tuple[str, nn.Linear]] = []
    for fqn, mod in model.named_modules():
        if filter_fn(mod, fqn):
            targets.append((fqn, mod))

    for fqn, linear in targets:
        q_mod = Int8DynamicLinear.from_linear(linear, compute_dtype=compute_dtype)
        _set_submodule(model, fqn, q_mod)

    logger.info(
        "[ACE-Step 1.5] Replaced %d linear layers with Int8DynamicLinear (%s).",
        len(targets),
        scope,
    )
    return len(targets)


def apply_int8_dynamic_quantization(
    model: nn.Module,
    scope: str,
    compute_dtype: torch.dtype,
    prefer_torchao: bool = True,
) -> str:
    """Apply INT8 dynamic quantization to a DiT or LM model and return the backend tag."""
    if prefer_torchao:
        ok, tag = _try_torchao_quantize(model, scope=scope)
        if ok:
            return tag
    _apply_int8_linear_fallback(model, scope=scope, compute_dtype=compute_dtype)
    return "int8_dynamic_linear"


def get_quantized_save_dir(
    source_dir: Path,
    save_suffix: str,
    target_root: Optional[Path] = None,
) -> Path:
    """Compute the quantized output directory preserving the base model name.

    When ``target_root`` is provided (e.g. the primary ``extra_model_paths.yaml`` directory),
    new quantized folders are written under ``target_root``; otherwise next to ``source_dir``.
    """
    suffix = (save_suffix or "_int8_convrot").strip()
    if source_dir.name.endswith(suffix):
        return source_dir
    base_root = target_root if target_root is not None else source_dir.parent
    return base_root / f"{source_dir.name}{suffix}"


def is_quantized_checkpoint_dir(path: Path) -> bool:
    """Return True if ``path`` contains a saved INT8 safetensors + scales checkpoint."""
    return (
        path.is_dir()
        and (path / QUANT_META_FILENAME).is_file()
        and (path / QUANT_WEIGHTS_FILENAME).is_file()
    )


def verify_quantized_checkpoint_complete(quant_dir: Path, scope: str) -> Tuple[bool, str]:
    """Verify that a saved INT8 checkpoint directory contains all required files before cleanup."""
    if not is_quantized_checkpoint_dir(quant_dir):
        return False, f"Missing {QUANT_META_FILENAME} or {QUANT_WEIGHTS_FILENAME}"

    weights_file = quant_dir / QUANT_WEIGHTS_FILENAME
    try:
        if weights_file.stat().st_size <= 0:
            return False, f"{QUANT_WEIGHTS_FILENAME} is empty"
    except OSError as exc:
        return False, f"Cannot stat {QUANT_WEIGHTS_FILENAME}: {exc}"

    try:
        meta = json.loads((quant_dir / QUANT_META_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or not meta.get("quantized_layers"):
            return False, f"{QUANT_META_FILENAME} contains no quantized_layers metadata"
    except Exception as exc:
        return False, f"Invalid {QUANT_META_FILENAME}: {exc}"

    if not (quant_dir / "config.json").is_file():
        return False, "Missing config.json in quantized directory"

    if scope == "dit" and not (quant_dir / "silence_latent.pt").is_file():
        return False, "Missing silence_latent.pt in quantized DiT directory"

    if scope == "lm":
        has_tokenizer = (
            (quant_dir / "tokenizer.json").is_file()
            or (quant_dir / "tokenizer_config.json").is_file()
        )
        if not has_tokenizer:
            return False, "Missing tokenizer files in quantized LM directory"

    return True, "ok"


def safe_delete_original_checkpoint(source_dir: Path, quantized_dir: Path, scope: str) -> bool:
    """Safely remove the original full-precision checkpoint folder after verifying the INT8 export.

    Refuses to delete anything outside the registered ACE-Step model root directories
    (``extra_model_paths.yaml`` paths or ``ComfyUI/models/acestep``), refuses to delete if
    ``source_dir == quantized_dir`` or if ``source_dir`` is itself a quantized folder, and
    only proceeds once ``verify_quantized_checkpoint_complete`` passes completely.
    """
    ok, reason = verify_quantized_checkpoint_complete(quantized_dir, scope=scope)
    if not ok:
        logger.warning(
            "[ACE-Step 1.5] Skipping deletion of original checkpoint '%s': quantized export '%s' failed verification (%s).",
            source_dir,
            quantized_dir,
            reason,
        )
        return False

    try:
        src_resolved = source_dir.resolve()
        q_resolved = quantized_dir.resolve()
        allowed_roots = {
            _norm_path_key(root.resolve())
            for root in (get_acestep_search_dirs() + [get_acestep_models_dir(create=False)])
        }
    except OSError as exc:
        logger.warning("[ACE-Step 1.5] Skipping deletion of '%s': path resolution failed (%s).", source_dir, exc)
        return False

    if src_resolved == q_resolved:
        return False
    if not src_resolved.is_dir():
        return False
    if is_quantized_checkpoint_dir(src_resolved):
        logger.warning("[ACE-Step 1.5] Refusing to delete quantized checkpoint directory '%s'.", src_resolved)
        return False
    if _norm_path_key(src_resolved.parent) not in allowed_roots:
        logger.warning(
            "[ACE-Step 1.5] Refusing to delete original checkpoint '%s' because it is outside registered ACE-Step model directories.",
            src_resolved,
        )
        return False

    gc.collect()
    try:
        shutil.rmtree(src_resolved)
        logger.info(
            "[ACE-Step 1.5] Deleted original checkpoint '%s' after verifying INT8 export at '%s'.",
            src_resolved,
            q_resolved,
        )
        return True
    except OSError as exc:
        logger.warning(
            "[ACE-Step 1.5] Could not automatically delete original checkpoint '%s' (%s). "
            "You can safely remove it manually; the INT8 checkpoint at '%s' is ready.",
            src_resolved,
            exc,
            q_resolved,
        )
        return False


def resolve_or_download_model_dir(
    model_name: str,
    *,
    category: str,
    precision: str,
    save_suffix: str,
    overwrite_quantized: bool = False,
    auto_download: bool = True,
) -> Tuple[Path, bool]:
    """Resolve a local cached INT8 checkpoint BEFORE ever touching or downloading the original model.

    Searches all configured ACE-Step directories (prioritizing ``extra_model_paths.yaml``):
    1. If ``model_name`` directly points to a valid INT8 quantized directory (e.g. ``*_int8_convrot``),
       auto-detect and return ``(requested_dir, True)`` regardless of ``precision`` (never downloads).
    2. If ``precision == "int8_dynamic"`` and ``overwrite_quantized`` is False, check for
       ``<model_name><save_suffix>`` across all registered ACE-Step directories FIRST. If it exists
       and is valid, return ``(candidate_qdir, True)`` immediately without downloading the original.
    3. Otherwise, ensure the base checkpoint is available locally (or download into the primary
       ``extra_model_paths.yaml`` directory via ``auto_download``) and return
       ``(source_dir, is_quantized_checkpoint_dir(source_dir))``.
    """
    requested_dir = resolve_checkpoint_path(model_name)

    # 1. Direct selection of a pre-quantized INT8 directory (auto-detect even if precision is bf16/fp16)
    if is_quantized_checkpoint_dir(requested_dir):
        if precision != "int8_dynamic":
            logger.info(
                "[ACE-Step 1.5] Auto-detected pre-quantized INT8 %s checkpoint '%s' "
                "(overriding precision='%s' -> 'int8_dynamic').",
                category,
                requested_dir.name,
                precision,
            )
        if overwrite_quantized:
            logger.warning(
                "[ACE-Step 1.5] overwrite_quantized=True ignored because '%s' is already a quantized folder. "
                "Select the base model name to re-quantize from full precision.",
                requested_dir.name,
            )
        return requested_dir, True

    # 2. Cache-first lookup for sibling quantized directory across ALL search dirs BEFORE downloading original
    if precision == "int8_dynamic" and not overwrite_quantized:
        candidates: List[Path] = []
        raw_candidate = Path(model_name).expanduser()
        if raw_candidate.is_absolute():
            candidates.append(get_quantized_save_dir(raw_candidate, save_suffix))
        else:
            for root in get_acestep_search_dirs():
                candidates.append(get_quantized_save_dir(root / model_name, save_suffix))
            sibling = get_quantized_save_dir(requested_dir, save_suffix)
            if sibling not in candidates:
                candidates.append(sibling)

        for candidate_qdir in candidates:
            if is_quantized_checkpoint_dir(candidate_qdir):
                logger.info(
                    "[ACE-Step 1.5] Reusing cached INT8 %s checkpoint '%s' (skipping original model lookup/download).",
                    category,
                    candidate_qdir,
                )
                return candidate_qdir, True

    # 3. No usable cached INT8 checkpoint (or full precision / explicit overwrite requested)
    source_dir = ensure_component_available(model_name, category=category, auto_download=auto_download)
    return source_dir, is_quantized_checkpoint_dir(source_dir)


def _extract_torchao_int8_and_scale(weight_param: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract raw ``(int8_data, float32_scale)`` tensors from a torchao AffineQuantizedTensor."""
    raw = weight_param.data if isinstance(weight_param, nn.Parameter) else weight_param

    # Path 1: torchao AffineQuantizedTensor with tensor_impl
    impl = getattr(raw, "tensor_impl", None)
    if impl is not None:
        int_data = getattr(impl, "int_data", None)
        scale = getattr(impl, "scale", None)
        if isinstance(int_data, torch.Tensor) and isinstance(scale, torch.Tensor) and int_data.dtype == torch.int8:
            scale_t = scale.detach().cpu().to(torch.float32)
            if scale_t.dim() == 1:
                scale_t = scale_t.unsqueeze(1)
            return int_data.detach().cpu().contiguous(), scale_t.contiguous()

    # Path 2: newer torchao Int8Tensor (qdata / scale)
    qdata = getattr(raw, "qdata", None)
    scale = getattr(raw, "scale", None)
    if isinstance(qdata, torch.Tensor) and isinstance(scale, torch.Tensor) and qdata.dtype == torch.int8:
        scale_t = scale.detach().cpu().to(torch.float32)
        if scale_t.dim() == 1:
            scale_t = scale_t.unsqueeze(1)
        return qdata.detach().cpu().contiguous(), scale_t.contiguous()

    # Path 3: dequantize and re-pack symmetrically if subclass exposes dequantize()
    if hasattr(raw, "dequantize"):
        try:
            fp = raw.dequantize().detach().cpu().to(torch.float32)
            max_abs = fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale_t = max_abs / 127.0
            q_w = torch.clamp(torch.round(fp / scale_t), -127, 127).to(torch.int8)
            return q_w.contiguous(), scale_t.contiguous()
        except Exception:
            return None

    return None


def _copy_checkpoint_sidecar_files(source_dir: Path, target_dir: Path) -> None:
    """Copy config, tokenizer, silence_latent, and custom remote-code Python files."""
    if not source_dir.is_dir() or source_dir.resolve() == target_dir.resolve():
        return

    skip_exts = {".safetensors", ".bin", ".pth", ".ckpt"}
    for item in source_dir.iterdir():
        if item.is_dir():
            continue
        if item.suffix.lower() in skip_exts:
            continue
        if item.name in (QUANT_META_FILENAME, QUANT_WEIGHTS_FILENAME):
            continue
        # Copy config.json, tokenizer files, modeling_*.py, configuration_*.py, silence_latent.pt, etc.
        dst = target_dir / item.name
        if not dst.exists() or item.name in ("config.json", "silence_latent.pt"):
            shutil.copy2(item, dst)


def _rematerialize_meta_buffers(model: nn.Module) -> None:
    """Reconstruct any non-persistent buffers left on ``meta`` device after ``assign=True`` load.

    Rebuilds ResidualFSQ/FSQ codebook/level buffers and RoPE ``inv_freq`` accurately instead
    of filling them with zeros, falling back to zero-init only for generic masks/caches.
    """
    for fqn, mod in model.named_modules():
        meta_bufs = [
            (bname, buf)
            for bname, buf in list(mod._buffers.items())
            if buf is not None and buf.device.type == "meta"
        ]
        if not meta_bufs:
            continue

        # 1. Reconstruct FSQ / ResidualFSQ non-persistent buffers accurately
        if _is_fsq_module(mod):
            levels_attr = getattr(mod, "levels", None)
            if levels_attr is not None:
                levels_list = [int(x) for x in levels_attr]
                levels_t = torch.tensor(levels_list, dtype=torch.int32, device="cpu")
                if "_levels" in mod._buffers and mod._buffers["_levels"] is not None and mod._buffers["_levels"].device.type == "meta":
                    mod._buffers["_levels"] = levels_t
                if "_basis" in mod._buffers and mod._buffers["_basis"] is not None and mod._buffers["_basis"].device.type == "meta":
                    basis_dtype = mod._buffers["_basis"].dtype
                    basis_vals = [1]
                    for lv in levels_list[:-1]:
                        basis_vals.append(basis_vals[-1] * lv)
                    mod._buffers["_basis"] = torch.tensor(basis_vals, dtype=basis_dtype, device="cpu")
                if (
                    "implicit_codebook" in mod._buffers
                    and mod._buffers["implicit_codebook"] is not None
                    and mod._buffers["implicit_codebook"].device.type == "meta"
                    and hasattr(mod, "_indices_to_codes")
                ):
                    try:
                        cb_size = int(getattr(mod, "codebook_size", 0) or math.prod(levels_list))
                        all_idx = torch.arange(cb_size, device="cpu")
                        cb = mod._indices_to_codes(all_idx).to(dtype=torch.float32, device="cpu")
                        mod._buffers["implicit_codebook"] = cb
                    except Exception:
                        pass

            if "scales" in mod._buffers and mod._buffers["scales"] is not None and mod._buffers["scales"].device.type == "meta":
                scales_shape = mod._buffers["scales"].shape
                num_q = scales_shape[0] if len(scales_shape) >= 1 else int(getattr(mod, "num_quantizers", 1))
                levels_ref = getattr(mod, "levels", None)
                if levels_ref is not None and len(scales_shape) >= 2:
                    base_lv = torch.tensor([float(x) for x in levels_ref], dtype=torch.float32, device="cpu")
                    scales_list = [base_lv ** (-i) for i in range(num_q)]
                    mod._buffers["scales"] = torch.stack(scales_list, dim=0).reshape(scales_shape)
                else:
                    mod._buffers["scales"] = torch.ones(scales_shape, dtype=torch.float32, device="cpu")

        # 2. Reconstruct RotaryEmbedding inv_freq accurately
        if "inv_freq" in mod._buffers and mod._buffers["inv_freq"] is not None and mod._buffers["inv_freq"].device.type == "meta":
            inv_shape = mod._buffers["inv_freq"].shape
            rebuilt = None
            rope_init_fn = getattr(mod, "rope_init_fn", None)
            mod_cfg = getattr(mod, "config", None)
            if callable(rope_init_fn) and mod_cfg is not None:
                try:
                    inv_f, _ = rope_init_fn(mod_cfg, "cpu")
                    if isinstance(inv_f, torch.Tensor) and inv_f.shape == inv_shape:
                        rebuilt = inv_f.to(device="cpu", dtype=torch.float32)
                except Exception:
                    rebuilt = None
            if rebuilt is None and len(inv_shape) == 1:
                dim = int(inv_shape[0]) * 2
                theta = float(
                    getattr(mod, "base", None)
                    or getattr(getattr(mod, "config", None), "rope_theta", None)
                    or 10000.0
                )
                rebuilt = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / float(dim)))
            if rebuilt is not None:
                mod._buffers["inv_freq"] = rebuilt

        # 3. Fallback for any remaining generic meta buffer
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                logger.debug("[ACE-Step 1.5] Zero-initializing remaining meta buffer '%s.%s'.", fqn, bname)
                mod._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")


def save_quantized_safetensors_checkpoint(
    model: nn.Module,
    source_dir: Path,
    output_dir: Path,
    *,
    scope: str,
    quant_backend: str,
    compute_dtype: Optional[torch.dtype] = None,
    overwrite: bool = False,
) -> Path:
    """Serialize a quantized model into ``<output_dir>/model_int8.safetensors`` + ``quant_meta.json``.

    Stores quantized linear weights as ``<fqn>.weight_int8`` (`int8`) and
    ``<fqn>.weight_scale`` (`float32`), normalizes non-quantized floating parameters
    and ordinary buffers to ``compute_dtype`` (if specified), and preserves
    precision-sensitive buffers (RoPE ``inv_freq``, ResidualFSQ ``implicit_codebook``/``scales``)
    in ``float32`` so the saved directory reloads cleanly without dtype clashes.
    """
    from safetensors.torch import save_file

    if is_quantized_checkpoint_dir(output_dir) and not overwrite:
        logger.info(
            "[ACE-Step 1.5] Quantized checkpoint already exists at '%s' (overwrite=False); skipping save.",
            output_dir,
        )
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint_sidecar_files(source_dir, output_dir)

    tensors: Dict[str, torch.Tensor] = {}
    quantized_layers: Dict[str, Dict[str, Any]] = {}
    handled_param_keys: set[str] = set()
    handled_buffer_keys: set[str] = set()

    def _export_fp_tensor(t: torch.Tensor, keep_fp32: bool = False) -> torch.Tensor:
        out = t.detach().cpu()
        if out.is_floating_point():
            if keep_fp32:
                out = out.to(torch.float32)
            elif compute_dtype is not None and out.dtype != compute_dtype:
                out = out.to(compute_dtype)
        return out.contiguous()

    # 1. Extract all quantized linear modules (Int8DynamicLinear or torchao-wrapped nn.Linear)
    for fqn, mod in model.named_modules():
        if isinstance(mod, Int8DynamicLinear):
            tensors[f"{fqn}.weight_int8"] = mod.weight_int8.detach().cpu().to(torch.int8).contiguous()
            tensors[f"{fqn}.weight_scale"] = mod.weight_scale.detach().cpu().to(torch.float32).contiguous()
            handled_buffer_keys.add(f"{fqn}.weight_int8")
            handled_buffer_keys.add(f"{fqn}.weight_scale")
            has_bias = mod.bias is not None
            if has_bias and mod.bias is not None:
                tensors[f"{fqn}.bias"] = _export_fp_tensor(mod.bias)
                handled_param_keys.add(f"{fqn}.bias")
            quantized_layers[fqn] = {
                "in_features": int(mod.in_features),
                "out_features": int(mod.out_features),
                "has_bias": has_bias,
            }
        elif isinstance(mod, nn.Linear) and _is_quantized_subclass(mod.weight.data):
            extracted = _extract_torchao_int8_and_scale(mod.weight)
            if extracted is not None:
                q_w, scale = extracted
                tensors[f"{fqn}.weight_int8"] = q_w
                tensors[f"{fqn}.weight_scale"] = scale
                handled_param_keys.add(f"{fqn}.weight")
                has_bias = mod.bias is not None
                if has_bias and mod.bias is not None:
                    tensors[f"{fqn}.bias"] = _export_fp_tensor(mod.bias)
                    handled_param_keys.add(f"{fqn}.bias")
                quantized_layers[fqn] = {
                    "in_features": int(mod.in_features),
                    "out_features": int(mod.out_features),
                    "has_bias": has_bias,
                }

    # 2. Collect all remaining non-quantized parameters and buffers
    for name, param in model.named_parameters():
        if name in handled_param_keys or param is None:
            continue
        if _is_quantized_subclass(param.data):
            extracted = _extract_torchao_int8_and_scale(param)
            if extracted is not None:
                q_w, scale = extracted
                tensors[f"{name}_int8"] = q_w
                tensors[f"{name}_scale"] = scale
                continue
        tensors[name] = _export_fp_tensor(param)

    for fqn, mod in model.named_modules():
        for bname, buf in mod._buffers.items():
            full_name = f"{fqn}.{bname}" if fqn else bname
            if full_name in handled_buffer_keys or buf is None:
                continue
            handled_buffer_keys.add(full_name)
            keep_fp32 = _is_precision_sensitive_fp32_buffer(mod, bname)
            tensors[full_name] = _export_fp_tensor(buf, keep_fp32=keep_fp32)

    # Break any shared tensor storage aliases (required by safetensors)
    seen_ptrs: set[int] = set()
    for key in list(tensors.keys()):
        t = tensors[key]
        ptr = t.data_ptr()
        if ptr in seen_ptrs:
            tensors[key] = t.clone()
        else:
            seen_ptrs.add(ptr)

    weights_path = output_dir / QUANT_WEIGHTS_FILENAME
    meta_path = output_dir / QUANT_META_FILENAME

    meta = {
        "format": "acestep_int8_safetensors_v1",
        "scope": scope,
        "quant_backend": quant_backend,
        "compute_dtype": str(compute_dtype).replace("torch.", "") if compute_dtype is not None else "auto",
        "source_model": source_dir.name,
        "weights_file": QUANT_WEIGHTS_FILENAME,
        "quantized_layer_count": len(quantized_layers),
        "quantized_layers": quantized_layers,
    }

    save_file(tensors, str(weights_path), metadata={"format": "acestep_int8_safetensors_v1", "scope": scope})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Free serialized CPU copies promptly
    del tensors
    gc.collect()

    logger.info(
        "[ACE-Step 1.5] Saved INT8 quantized checkpoint (%d quantized layers, compute_dtype=%s) to '%s'.",
        len(quantized_layers),
        meta["compute_dtype"],
        output_dir,
    )
    return output_dir


def load_quantized_safetensors_model(
    model_dir: Path,
    *,
    scope: str,
    compute_dtype: torch.dtype,
    attn_implementation: Optional[str] = None,
) -> nn.Module:
    """Load a previously saved INT8 safetensors + scales checkpoint directly without re-quantizing.

    Uses ``accelerate.init_empty_weights`` (or meta device) to instantiate the
    model architecture with zero BF16 memory allocation, replaces quantized
    linear layers with ``Int8DynamicLinear``, loads ``model_int8.safetensors``
    directly into CPU RAM, reconstructs any missing non-persistent meta buffers
    accurately, and aligns all dtypes coherently (preserving FP32 FSQ/RoPE/scales).
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    meta = json.loads((model_dir / QUANT_META_FILENAME).read_text(encoding="utf-8"))
    quantized_layers: Dict[str, Dict[str, Any]] = meta.get("quantized_layers", {})
    weights_file = model_dir / meta.get("weights_file", QUANT_WEIGHTS_FILENAME)

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if attn_implementation is not None:
        try:
            config._attn_implementation = attn_implementation
        except Exception:
            pass

    model_cls = AutoModelForCausalLM if scope == "lm" else AutoModel

    # Build skeleton on meta device so we never materialize a full BF16/FP32 copy in RAM
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = model_cls.from_config(config, trust_remote_code=True)
    except Exception:
        with torch.device("meta"):
            model = model_cls.from_config(config, trust_remote_code=True)

    # Swap quantized linear modules before loading state dict
    for fqn, spec in quantized_layers.items():
        q_mod = Int8DynamicLinear(
            in_features=int(spec["in_features"]),
            out_features=int(spec["out_features"]),
            has_bias=bool(spec.get("has_bias", True)),
            compute_dtype=compute_dtype,
        )
        _set_submodule(model, fqn, q_mod)

    state_dict = load_file(str(weights_file), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    gc.collect()

    # Re-materialize any remaining non-persistent meta buffers (RoPE inv_freq, FSQ codebooks/levels)
    _rematerialize_meta_buffers(model)

    # Restore tied weights if LM ties embed_tokens & lm_head
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    # Align all non-quantized params & ordinary buffers to compute_dtype while preserving
    # FP32 precision-sensitive buffers (weight_scale, inv_freq, FSQ implicit_codebook/scales)
    # and installing ResidualFSQ boundary dtype guards.
    align_module_dtypes(model, compute_dtype)

    if missing:
        logger.debug("[ACE-Step 1.5] Quantized load missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.debug("[ACE-Step 1.5] Quantized load unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    model.eval()
    logger.info(
        "[ACE-Step 1.5] Directly loaded cached INT8 checkpoint '%s' (%d quantized layers, compute_dtype=%s).",
        model_dir.name,
        len(quantized_layers),
        compute_dtype,
    )
    return model


# ==============================================================================
# Stage Model Handles (Lazy Re-hydration + Deterministic Unload)
# ==============================================================================


@dataclass
class AceStepLMHandle:
    """Lightweight handle for the 5Hz Language Model stage (``ACESTEP_LM``)."""

    model_name: str
    checkpoint_dir: Path
    resolved_model_dir: Path
    device_mode: str = "auto_offload"  # "cuda" | "cpu" | "auto_offload"
    precision: str = "bf16"            # "bf16" | "fp16" | "int8_dynamic"
    save_quantized: bool = False
    save_suffix: str = "_int8_convrot"
    overwrite_quantized: bool = False
    delete_original_after_quantize: bool = False
    auto_download: bool = True
    quant_backend: str = "none"
    handler: Any = field(default=None, repr=False)
    _defer_original_delete: bool = field(default=False, repr=False)
    _last_quantized_source_dir: Optional[Path] = field(default=None, repr=False)
    _export_compute_dtype: Optional[torch.dtype] = field(default=None, repr=False)

    def ensure_loaded(self) -> Any:
        """Load or move the ``LLMHandler`` into memory ready for PyTorch (``pt``) inference."""
        _ensure_acestep_package()
        from acestep.llm_inference import LLMHandler
        from transformers import AutoTokenizer

        exec_device = get_compute_device("cpu" if self.device_mode == "cpu" else "cuda")
        effective_precision = (
            "int8_dynamic"
            if (self.precision == "int8_dynamic" or self.quant_backend != "none")
            else self.precision
        )
        compute_dtype = resolve_torch_dtype(effective_precision, exec_device)

        if self.handler is not None and getattr(self.handler, "llm_initialized", False):
            target_dev = "cpu" if self.device_mode == "cpu" else str(exec_device)
            if self.handler.llm is not None:
                if target_dev != "cpu":
                    request_vram_headroom(5.0 if effective_precision == "int8_dynamic" else 9.0, exec_device)
                move_module_to_device_safe(self.handler.llm, target_dev, compute_dtype)
                self.handler.device = target_dev
                self.handler.dtype = compute_dtype
            return self.handler

        # Check whether the handle already points to a cached INT8 checkpoint before touching disk/HF
        if (
            self.resolved_model_dir is not None
            and is_quantized_checkpoint_dir(self.resolved_model_dir)
            and (
                not self.overwrite_quantized
                or self.resolved_model_dir == resolve_checkpoint_path(self.model_name)
            )
        ):
            load_dir = self.resolved_model_dir
            source_dir = self.resolved_model_dir
            is_cached_int8 = True
            effective_precision = "int8_dynamic"
        else:
            load_dir, is_cached_int8 = resolve_or_download_model_dir(
                self.model_name,
                category="lm",
                precision=self.precision,
                save_suffix=self.save_suffix,
                overwrite_quantized=self.overwrite_quantized,
                auto_download=self.auto_download,
            )
            source_dir = load_dir
            if is_cached_int8:
                effective_precision = "int8_dynamic"

        compute_dtype = resolve_torch_dtype(effective_precision, exec_device)
        export_dtype = self._export_compute_dtype or compute_dtype

        if exec_device.type == "cuda":
            est_gb = 5.0 if effective_precision == "int8_dynamic" else 9.0
            request_vram_headroom(est_gb, exec_device)

        self.resolved_model_dir = load_dir
        llm_handler = LLMHandler()

        # Always use PyTorch ("pt") backend for Python 3.13 & strict VRAM cleanup compatibility.
        if is_cached_int8:
            tokenizer = AutoTokenizer.from_pretrained(str(load_dir), use_fast=True, trust_remote_code=True)
            model = load_quantized_safetensors_model(
                load_dir,
                scope="lm",
                compute_dtype=compute_dtype,
            )
            self.quant_backend = "cached_int8_safetensors"
            target_dev = "cpu" if self.device_mode == "cpu" else str(exec_device)
            move_module_to_device_safe(model, target_dev, compute_dtype)
            llm_handler.llm = model
            llm_handler.llm_tokenizer = tokenizer
            llm_handler._lm_full_model_path = str(load_dir)
            llm_handler.llm_backend = "pt"
            llm_handler.device = target_dev
            llm_handler.dtype = compute_dtype
            llm_handler.offload_to_cpu = self.device_mode == "auto_offload"
            llm_handler.llm_initialized = True
        else:
            # Initialize via official LLMHandler.initialize() with backend="pt"
            init_device = "cpu" if self.device_mode == "cpu" else str(exec_device)
            effective_ckpt_dir = load_dir.parent if load_dir.is_dir() else self.checkpoint_dir
            lm_model_arg = load_dir.name if load_dir.is_dir() else str(load_dir)
            status, ok = llm_handler.initialize(
                checkpoint_dir=str(effective_ckpt_dir),
                lm_model_path=lm_model_arg,
                backend="pt",
                device=init_device if effective_precision != "int8_dynamic" else "cpu",
                offload_to_cpu=(self.device_mode == "auto_offload"),
                dtype=compute_dtype,
            )
            if not ok:
                raise RuntimeError(f"Failed to initialize ACE-Step 5Hz LM ({self.model_name}): {status}")

            if llm_handler.llm is not None:
                align_module_dtypes(llm_handler.llm, compute_dtype)

            if effective_precision == "int8_dynamic" and llm_handler.llm is not None:
                self.quant_backend = apply_int8_dynamic_quantization(
                    llm_handler.llm,
                    scope="lm",
                    compute_dtype=export_dtype,
                    prefer_torchao=True,
                )
                if self.save_quantized:
                    out_dir = get_quantized_save_dir(
                        source_dir,
                        self.save_suffix,
                        target_root=get_acestep_models_dir(create=True),
                    )
                    save_quantized_safetensors_checkpoint(
                        llm_handler.llm,
                        source_dir=source_dir,
                        output_dir=out_dir,
                        scope="lm",
                        quant_backend=self.quant_backend,
                        compute_dtype=export_dtype,
                        overwrite=self.overwrite_quantized,
                    )
                    self.resolved_model_dir = out_dir
                    self.overwrite_quantized = False
                    self._last_quantized_source_dir = source_dir
                    if self.delete_original_after_quantize and not self._defer_original_delete:
                        safe_delete_original_checkpoint(source_dir, out_dir, scope="lm")

                target_dev = "cpu" if self.device_mode == "cpu" else str(exec_device)
                move_module_to_device_safe(llm_handler.llm, target_dev, compute_dtype)
                llm_handler.device = target_dev
                llm_handler.dtype = compute_dtype

        # Ensure constrained decoding processor is initialized and allows up to 600s
        # because our sequential ComfyUI nodes never co-locate LM and DiT in VRAM.
        if getattr(llm_handler, "llm_tokenizer", None) is not None:
            try:
                from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor
                if getattr(llm_handler, "constrained_processor", None) is None:
                    llm_handler.constrained_processor = MetadataConstrainedLogitsProcessor(
                        tokenizer=llm_handler.llm_tokenizer,
                        enabled=True,
                        debug=False,
                        max_duration=600,
                    )
                elif hasattr(llm_handler.constrained_processor, "max_duration"):
                    llm_handler.constrained_processor.max_duration = max(
                        600, int(getattr(llm_handler.constrained_processor, "max_duration", 600) or 600)
                    )
            except Exception as exc:
                logger.debug("[ACE-Step 1.5] Optional constrained processor setup skipped: %s", exc)

        self.handler = llm_handler
        return self.handler

    def unload(self, release_reference: bool = True) -> None:
        """Strictly unload the 5Hz LM from VRAM and purge CUDA/ComfyUI caches."""
        if self.handler is not None:
            llm_model = getattr(self.handler, "llm", None)
            if llm_model is not None:
                try:
                    move_module_to_device_safe(llm_model, "cpu")
                except Exception:
                    pass
            if release_reference:
                try:
                    if hasattr(self.handler, "unload"):
                        self.handler.unload()
                except Exception:
                    pass
                self.handler.llm = None
                self.handler.llm_tokenizer = None
                self.handler.llm_initialized = False
                self.handler = None
        purge_vram(aggressive=True)


@dataclass
class AceStepDiTHandle:
    """Lightweight handle for the DiT + Text Encoder stage (``ACESTEP_DIT``).

    Intentionally excludes the VAE so DiT 4B XL and AutoencoderOobleck never
    occupy VRAM simultaneously.
    """

    model_name: str
    checkpoint_dir: Path
    resolved_model_dir: Path
    precision: str = "bf16"               # "bf16" | "fp16" | "int8_dynamic"
    save_quantized: bool = False
    save_suffix: str = "_int8_convrot"
    overwrite_quantized: bool = False
    delete_original_after_quantize: bool = False
    offload_dit_to_cpu: bool = True
    vram_offload_threshold_gb: float = 1.5
    use_flash_attention: bool = True
    auto_download: bool = True
    quant_backend: str = "none"
    handler: Any = field(default=None, repr=False)
    _defer_original_delete: bool = field(default=False, repr=False)
    _last_quantized_source_dir: Optional[Path] = field(default=None, repr=False)

    def _select_attention_impl(self, handler: Any, device: torch.device) -> List[str]:
        """Build ordered attention implementation fallback list matching ACE-Step 1.5."""
        candidates: List[str] = []
        if self.use_flash_attention and hasattr(handler, "is_flash_attention_available"):
            try:
                if handler.is_flash_attention_available(str(device)):
                    candidates.append("flash_attention_2")
            except Exception:
                pass
        if device.type == "cuda":
            try:
                major, _ = torch.cuda.get_device_capability(device)
                if major < 8:
                    # Pre-Ampere GPUs avoid fp16 SDPA overflow by preferring eager
                    candidates.append("eager")
            except Exception:
                pass
        for fallback in ("sdpa", "eager"):
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def ensure_loaded(self) -> Any:
        """Load DiT + ``Qwen3-Embedding-0.6B`` + ``silence_latent.pt`` into an ``AceStepHandler``."""
        _ensure_acestep_package()
        from acestep.handler import AceStepHandler
        from transformers import AutoModel, AutoTokenizer

        exec_device = get_compute_device("cuda")
        effective_precision = (
            "int8_dynamic"
            if (self.precision == "int8_dynamic" or self.quant_backend != "none")
            else self.precision
        )
        compute_dtype = resolve_torch_dtype(effective_precision, exec_device)

        if (
            self.handler is not None
            and getattr(self.handler, "model", None) is not None
            and getattr(self.handler, "text_encoder", None) is not None
        ):
            self.handler.device = str(exec_device)
            self.handler.dtype = compute_dtype
            self.handler.offload_to_cpu = True
            self.handler.offload_dit_to_cpu = self.offload_dit_to_cpu
            align_module_dtypes(self.handler.model, compute_dtype)
            install_handler_safe_transfer(self.handler)
            return self.handler

        # Check whether the handle already points to a cached INT8 checkpoint before touching disk/HF
        if (
            self.resolved_model_dir is not None
            and is_quantized_checkpoint_dir(self.resolved_model_dir)
            and (
                not self.overwrite_quantized
                or self.resolved_model_dir == resolve_checkpoint_path(self.model_name)
            )
        ):
            load_dir = self.resolved_model_dir
            source_dir = self.resolved_model_dir
            is_cached_int8 = True
            effective_precision = "int8_dynamic"
        else:
            load_dir, is_cached_int8 = resolve_or_download_model_dir(
                self.model_name,
                category="dit",
                precision=self.precision,
                save_suffix=self.save_suffix,
                overwrite_quantized=self.overwrite_quantized,
                auto_download=self.auto_download,
            )
            source_dir = load_dir
            if is_cached_int8:
                effective_precision = "int8_dynamic"

        compute_dtype = resolve_torch_dtype(effective_precision, exec_device)

        if exec_device.type == "cuda":
            est_gb = 5.5 if effective_precision == "int8_dynamic" else 9.5
            request_vram_headroom(est_gb, exec_device)

        text_encoder_dir = ensure_text_encoder_available(auto_download=self.auto_download)
        self.resolved_model_dir = load_dir

        handler = AceStepHandler()
        handler.device = str(exec_device)
        handler.dtype = compute_dtype
        handler.offload_to_cpu = True
        handler.offload_dit_to_cpu = self.offload_dit_to_cpu
        handler.vae = None  # Explicitly excluded; loaded in AceStepVAELoader
        install_handler_safe_transfer(handler)

        attn_candidates = self._select_attention_impl(handler, exec_device)
        chosen_attn = attn_candidates[0]

        if is_cached_int8:
            model = load_quantized_safetensors_model(
                load_dir,
                scope="dit",
                compute_dtype=compute_dtype,
                attn_implementation=chosen_attn,
            )
            self.quant_backend = "cached_int8_safetensors"
        else:
            model = None
            last_err: Optional[Exception] = None
            for candidate in attn_candidates:
                try:
                    model = AutoModel.from_pretrained(
                        str(load_dir),
                        trust_remote_code=True,
                        attn_implementation=candidate,
                        dtype=compute_dtype,
                    )
                    chosen_attn = candidate
                    break
                except TypeError:
                    try:
                        model = AutoModel.from_pretrained(
                            str(load_dir),
                            trust_remote_code=True,
                            attn_implementation=candidate,
                            torch_dtype=compute_dtype,
                        )
                        chosen_attn = candidate
                        break
                    except Exception as exc:
                        last_err = exc
                except Exception as exc:
                    last_err = exc
            if model is None:
                raise RuntimeError(f"Failed to load DiT model from '{load_dir}': {last_err}") from last_err

            model.eval()
            # Ensure FP32 shards (e.g. acestep-v15-xl-base/sft/turbo) and all auxiliary modules
            # are coherently cast to compute_dtype while preserving FP32 FSQ/RoPE buffers.
            align_module_dtypes(model, compute_dtype)

            if effective_precision == "int8_dynamic":
                self.quant_backend = apply_int8_dynamic_quantization(
                    model,
                    scope="dit",
                    compute_dtype=compute_dtype,
                    prefer_torchao=True,
                )
                align_module_dtypes(model, compute_dtype)
                if self.save_quantized:
                    out_dir = get_quantized_save_dir(
                        source_dir,
                        self.save_suffix,
                        target_root=get_acestep_models_dir(create=True),
                    )
                    save_quantized_safetensors_checkpoint(
                        model,
                        source_dir=source_dir,
                        output_dir=out_dir,
                        scope="dit",
                        quant_backend=self.quant_backend,
                        compute_dtype=compute_dtype,
                        overwrite=self.overwrite_quantized,
                    )
                    self.resolved_model_dir = out_dir
                    self.overwrite_quantized = False
                    self._last_quantized_source_dir = source_dir
                    if self.delete_original_after_quantize and not self._defer_original_delete:
                        safe_delete_original_checkpoint(source_dir, out_dir, scope="dit")

        model.config._attn_implementation = chosen_attn
        align_module_dtypes(model, compute_dtype)
        handler.model = model
        handler.config = model.config

        if hasattr(handler, "_sync_alignment_config"):
            try:
                handler._sync_alignment_config()
            except Exception:
                pass
        if hasattr(handler, "_apply_cuda_bool_argsort_workaround"):
            try:
                handler._apply_cuda_bool_argsort_workaround()
            except Exception:
                pass

        # Load silence_latent.pt (from resolved dir or original source dir)
        silence_path = load_dir / "silence_latent.pt"
        if not silence_path.exists():
            silence_path = source_dir / "silence_latent.pt"
        if not silence_path.exists():
            raise FileNotFoundError(f"silence_latent.pt not found in '{load_dir}' or '{source_dir}'.")
        handler.silence_latent = torch.load(str(silence_path), weights_only=True).transpose(1, 2).to("cpu").to(compute_dtype)

        # Load Qwen3-Embedding-0.6B on CPU; AceStepSampler moves it transiently during text embedding
        handler.text_tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_dir))
        handler.text_encoder = AutoModel.from_pretrained(str(text_encoder_dir))
        move_module_to_device_safe(handler.text_encoder, "cpu", torch.float32)
        handler.text_encoder.eval()

        # Keep DiT on CPU until AceStepSampler explicitly starts diffusion
        move_module_to_device_safe(handler.model, "cpu", compute_dtype)
        handler.last_init_params = {
            "config_path": self.model_name,
            "device": str(exec_device),
            "precision": self.precision,
            "offload_dit_to_cpu": self.offload_dit_to_cpu,
        }

        self.handler = handler
        return self.handler

    def unload(self, release_reference: bool = True) -> None:
        """Strictly unload DiT, text encoder, and silence latent from VRAM."""
        if self.handler is not None:
            for attr in ("model", "text_encoder"):
                mod = getattr(self.handler, attr, None)
                if mod is not None:
                    try:
                        move_module_to_device_safe(mod, "cpu")
                    except Exception:
                        pass
            if getattr(self.handler, "silence_latent", None) is not None:
                try:
                    self.handler.silence_latent = self.handler.silence_latent.to("cpu")
                except Exception:
                    pass
            if release_reference:
                self.handler.model = None
                self.handler.text_encoder = None
                self.handler.text_tokenizer = None
                self.handler.silence_latent = None
                self.handler = None
        purge_vram(aggressive=True)


@dataclass
class AceStepVAEHandle:
    """Lightweight handle for the AutoencoderOobleck VAE stage (``ACESTEP_VAE``)."""

    model_name: str
    checkpoint_dir: Path
    resolved_model_dir: Path
    device_mode: str = "cuda"   # "cuda" | "cpu"
    precision: str = "fp16"     # "bf16" | "fp16" | "fp32"
    auto_download: bool = True
    sample_rate: int = 48000
    vae: Any = field(default=None, repr=False)

    def ensure_loaded(self) -> Tuple[Any, torch.device, torch.dtype]:
        """Load ``AutoencoderOobleck`` onto the requested device with safe precision."""
        from diffusers.models import AutoencoderOobleck

        exec_device = get_compute_device("cpu" if self.device_mode == "cpu" else "cuda")
        vae_dtype = resolve_torch_dtype(self.precision, exec_device)
        if exec_device.type == "cpu":
            vae_dtype = torch.float32

        if self.vae is not None:
            if exec_device.type == "cuda":
                request_vram_headroom(1.5, exec_device)
            move_module_to_device_safe(self.vae, exec_device, vae_dtype)
            self.vae.eval()
            return self.vae, exec_device, vae_dtype

        if exec_device.type == "cuda":
            request_vram_headroom(1.5, exec_device)

        vae_dir = ensure_component_available(self.model_name, category="vae", auto_download=self.auto_download)
        self.resolved_model_dir = vae_dir

        vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
        move_module_to_device_safe(vae, exec_device, vae_dtype)
        vae.eval()
        self.vae = vae
        return self.vae, exec_device, vae_dtype

    def unload(self, release_reference: bool = True) -> None:
        """Strictly unload VAE from VRAM and purge CUDA/ComfyUI caches."""
        if self.vae is not None:
            try:
                move_module_to_device_safe(self.vae, "cpu", torch.float32)
            except Exception:
                pass
            if release_reference:
                self.vae = None
        purge_vram(aggressive=True)
