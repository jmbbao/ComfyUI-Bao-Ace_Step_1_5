"""ComfyUI Loader nodes for ACE-Step 1.5 (5Hz LM, DiT 2B/XL 4B, and AutoencoderOobleck VAE).

To guarantee that a 12 GB VRAM GPU can execute 4B XL DiT + 4B 5Hz LM without
Out-Of-Memory errors, each loader creates a lightweight, lazy stage handle
(``ACESTEP_LM``, ``ACESTEP_DIT``, ``ACESTEP_VAE``). Weights are only transferred
to CUDA during their own execution node and are immediately purged afterwards,
preventing ComfyUI's loader execution order from co-locating multiple large
models in VRAM simultaneously.
"""

from __future__ import annotations

import logging
from typing import Tuple

from .memory_utils import (
    AceStepDiTHandle,
    AceStepLMHandle,
    AceStepVAEHandle,
    ensure_component_available,
    ensure_text_encoder_available,
    get_acestep_models_dir,
    is_quantized_checkpoint_dir,
    list_checkpoint_choices,
    purge_vram,
    resolve_or_download_model_dir,
    safe_delete_original_checkpoint,
)

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")


class AceStepLMLoader:
    """Configure and prepare an ACE-Step 1.5 5Hz Language Model (0.6B / 1.7B / 4B)."""

    CATEGORY = "ACE-Step 1.5/Loaders"
    RETURN_TYPES = ("ACESTEP_LM",)
    RETURN_NAMES = ("lm_model",)
    FUNCTION = "load_lm"

    @classmethod
    def INPUT_TYPES(cls):
        lm_choices = list_checkpoint_choices("lm")
        default_lm = "acestep-5Hz-lm-1.7B" if "acestep-5Hz-lm-1.7B" in lm_choices else lm_choices[0]
        return {
            "required": {
                "model_path": (
                    lm_choices,
                    {
                        "default": default_lm,
                        "tooltip": (
                            "5Hz Language Model checkpoint inside ComfyUI/models/acestep/ "
                            "(e.g. acestep-5Hz-lm-0.6B, 1.7B, or 4B). Local quantized folders "
                            "are detected automatically."
                        ),
                    },
                ),
                "device": (
                    ["auto_offload", "cuda", "cpu"],
                    {
                        "default": "auto_offload",
                        "tooltip": (
                            "Execution device for the PyTorch ('pt') LM backend. "
                            "'auto_offload' keeps VRAM minimal for 12 GB GPUs; 'cpu' runs "
                            "entirely in system RAM with zero VRAM cost."
                        ),
                    },
                ),
                "precision": (
                    ["bf16", "fp16", "int8_dynamic"],
                    {
                        "default": "int8_dynamic",
                        "tooltip": (
                            "Weight/compute precision. 'int8_dynamic' uses torchao (or native "
                            "PyTorch fallback) to shrink the 4B LM from ~8 GB to ~4.5 GB VRAM."
                        ),
                    },
                ),
                "save_quantized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When precision is 'int8_dynamic', export quantized INT8 weights "
                            "and per-channel scales to a sibling .safetensors folder for instant "
                            "reloading on future runs."
                        ),
                    },
                ),
                "save_suffix": (
                    "STRING",
                    {
                        "default": "_int8_convrot",
                        "multiline": False,
                        "tooltip": (
                            "Directory suffix appended to the base checkpoint name when saving "
                            "or discovering cached INT8 checkpoints (e.g. acestep-5Hz-lm-4B_int8_convrot)."
                        ),
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Automatically download missing official checkpoints from Hugging Face "
                            "into ComfyUI/models/acestep/."
                        ),
                    },
                ),
            },
            "optional": {
                "overwrite_quantized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If False (default), existing quantized folders matching save_suffix "
                            "are loaded directly without re-quantizing or re-downloading the original model."
                        ),
                    },
                ),
                "delete_original_after_quantize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Optional disk-saving cleanup: when save_quantized=True finishes writing and "
                            "verifying the INT8 folder inside models/acestep/, automatically delete the "
                            "original full-precision checkpoint folder."
                        ),
                    },
                ),
            },
        }

    def load_lm(
        self,
        model_path: str,
        device: str,
        precision: str,
        save_quantized: bool,
        save_suffix: str,
        auto_download: bool,
        overwrite_quantized: bool = False,
        delete_original_after_quantize: bool = False,
    ) -> Tuple[AceStepLMHandle]:
        models_dir = get_acestep_models_dir()

        # Cache-first resolution: reuse existing *_int8_convrot or directly selected INT8
        # checkpoint BEFORE ever looking up or downloading the original full-precision model.
        resolved_dir, is_cached_int8 = resolve_or_download_model_dir(
            model_path,
            category="lm",
            precision=precision,
            save_suffix=save_suffix,
            overwrite_quantized=overwrite_quantized,
            auto_download=auto_download,
        )
        effective_precision = "int8_dynamic" if is_cached_int8 else precision

        handle = AceStepLMHandle(
            model_name=model_path,
            checkpoint_dir=models_dir,
            resolved_model_dir=resolved_dir,
            device_mode=device,
            precision=effective_precision,
            save_quantized=save_quantized,
            save_suffix=save_suffix,
            overwrite_quantized=overwrite_quantized if not is_cached_int8 else False,
            delete_original_after_quantize=delete_original_after_quantize,
            auto_download=auto_download,
        )

        # If the user explicitly asked to quantize & save an uncached checkpoint, perform the
        # quantization pass once on CPU and immediately release memory so VRAM stays free.
        if (
            effective_precision == "int8_dynamic"
            and save_quantized
            and not is_cached_int8
        ):
            logger.info("[ACE-Step 1.5] Pre-quantizing LM '%s' and saving to disk...", model_path)
            saved_device_mode = handle.device_mode
            handle.device_mode = "cpu"
            handle._defer_original_delete = True
            try:
                handle.ensure_loaded()
            finally:
                handle.unload(release_reference=True)
                handle.device_mode = saved_device_mode
                handle._defer_original_delete = False
                purge_vram(aggressive=True)

            # Delete original folder ONLY after unloading all references & verifying INT8 export
            if (
                delete_original_after_quantize
                and handle._last_quantized_source_dir is not None
                and is_quantized_checkpoint_dir(handle.resolved_model_dir)
            ):
                safe_delete_original_checkpoint(
                    handle._last_quantized_source_dir,
                    handle.resolved_model_dir,
                    scope="lm",
                )

        return (handle,)


class AceStepDiTLoader:
    """Configure and prepare an ACE-Step 1.5 Diffusion Transformer (2B or XL 4B)."""

    CATEGORY = "ACE-Step 1.5/Loaders"
    RETURN_TYPES = ("ACESTEP_DIT",)
    RETURN_NAMES = ("dit_model",)
    FUNCTION = "load_dit"

    @classmethod
    def INPUT_TYPES(cls):
        dit_choices = list_checkpoint_choices("dit")
        default_dit = "acestep-v15-xl-sft" if "acestep-v15-xl-sft" in dit_choices else dit_choices[0]
        return {
            "required": {
                "model_path": (
                    dit_choices,
                    {
                        "default": default_dit,
                        "tooltip": (
                            "DiT checkpoint folder in ComfyUI/models/acestep/ "
                            "(2B: base/sft/turbo; XL 4B: acestep-v15-xl-base/sft/turbo)."
                        ),
                    },
                ),
                "precision": (
                    ["bf16", "fp16", "int8_dynamic"],
                    {
                        "default": "int8_dynamic",
                        "tooltip": (
                            "DiT weight/compute precision. 'int8_dynamic' quantizes decoder "
                            "linear layers via torchao (excluding tokenizer/detokenizer), "
                            "reducing XL 4B from ~9 GB to ~5 GB VRAM."
                        ),
                    },
                ),
                "save_quantized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When precision is 'int8_dynamic', save quantized INT8 weights and "
                            "scales into a sibling .safetensors directory alongside config.json "
                            "and silence_latent.pt."
                        ),
                    },
                ),
                "save_suffix": (
                    "STRING",
                    {
                        "default": "_int8_convrot",
                        "multiline": False,
                        "tooltip": (
                            "Suffix appended to the base model name for cached INT8 checkpoints "
                            "(e.g. acestep-v15-xl-sft_int8_convrot). Existing folders with this "
                            "suffix are loaded directly without re-quantizing."
                        ),
                    },
                ),
                "offload_dit_to_cpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable dynamic block-level CPU offloading when free VRAM drops "
                            "below vram_offload_threshold_gb. Recommended for 12 GB GPUs "
                            "running XL 4B with long durations or CFG."
                        ),
                    },
                ),
                "vram_offload_threshold_gb": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.25,
                        "max": 24.0,
                        "step": 0.25,
                        "tooltip": (
                            "Minimum free VRAM headroom (GB) before activating block-by-block "
                            "CPU offloading during DiT diffusion."
                        ),
                    },
                ),
                "use_flash_attention": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use Flash Attention 2 when installed and supported; otherwise "
                            "automatically falls back to PyTorch SDPA or eager attention."
                        ),
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Automatically download missing DiT or Qwen3-Embedding-0.6B weights "
                            "from Hugging Face into ComfyUI/models/acestep/."
                        ),
                    },
                ),
            },
            "optional": {
                "overwrite_quantized": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If False (default), an existing quantized folder matching save_suffix "
                            "is reused directly without re-quantizing or re-downloading the original model."
                        ),
                    },
                ),
                "delete_original_after_quantize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Optional disk-saving cleanup: when save_quantized=True finishes writing and "
                            "verifying the INT8 folder inside models/acestep/, automatically delete the "
                            "original full-precision checkpoint folder."
                        ),
                    },
                ),
            },
        }

    def load_dit(
        self,
        model_path: str,
        precision: str,
        save_quantized: bool,
        save_suffix: str,
        offload_dit_to_cpu: bool,
        vram_offload_threshold_gb: float,
        use_flash_attention: bool,
        auto_download: bool,
        overwrite_quantized: bool = False,
        delete_original_after_quantize: bool = False,
    ) -> Tuple[AceStepDiTHandle]:
        models_dir = get_acestep_models_dir()

        # Cache-first resolution: reuse existing *_int8_convrot or directly selected INT8
        # checkpoint BEFORE ever looking up or downloading the original full-precision model.
        resolved_dir, is_cached_int8 = resolve_or_download_model_dir(
            model_path,
            category="dit",
            precision=precision,
            save_suffix=save_suffix,
            overwrite_quantized=overwrite_quantized,
            auto_download=auto_download,
        )
        # Downloads ONLY Qwen3-Embedding-0.6B/* if missing (never pulls extra base models).
        ensure_text_encoder_available(auto_download=auto_download)
        effective_precision = "int8_dynamic" if is_cached_int8 else precision

        handle = AceStepDiTHandle(
            model_name=model_path,
            checkpoint_dir=models_dir,
            resolved_model_dir=resolved_dir,
            precision=effective_precision,
            save_quantized=save_quantized,
            save_suffix=save_suffix,
            overwrite_quantized=overwrite_quantized if not is_cached_int8 else False,
            delete_original_after_quantize=delete_original_after_quantize,
            offload_dit_to_cpu=offload_dit_to_cpu,
            vram_offload_threshold_gb=vram_offload_threshold_gb,
            use_flash_attention=use_flash_attention,
            auto_download=auto_download,
        )

        if (
            effective_precision == "int8_dynamic"
            and save_quantized
            and not is_cached_int8
        ):
            logger.info("[ACE-Step 1.5] Pre-quantizing DiT '%s' and saving to disk...", model_path)
            handle._defer_original_delete = True
            try:
                handle.ensure_loaded()
            finally:
                handle.unload(release_reference=True)
                handle._defer_original_delete = False
                purge_vram(aggressive=True)

            # Delete original folder ONLY after unloading all references & verifying INT8 export
            if (
                delete_original_after_quantize
                and handle._last_quantized_source_dir is not None
                and is_quantized_checkpoint_dir(handle.resolved_model_dir)
            ):
                safe_delete_original_checkpoint(
                    handle._last_quantized_source_dir,
                    handle.resolved_model_dir,
                    scope="dit",
                )

        return (handle,)


class AceStepVAELoader:
    """Configure and prepare the ACE-Step 1.5 AutoencoderOobleck VAE (48 kHz stereo)."""

    CATEGORY = "ACE-Step 1.5/Loaders"
    RETURN_TYPES = ("ACESTEP_VAE",)
    RETURN_NAMES = ("vae_model",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        vae_choices = list_checkpoint_choices("vae")
        default_vae = "vae" if "vae" in vae_choices else vae_choices[0]
        return {
            "required": {
                "model_path": (
                    vae_choices,
                    {
                        "default": default_vae,
                        "tooltip": (
                            "VAE checkpoint folder inside ComfyUI/models/acestep/ "
                            "('vae' for official AutoencoderOobleck or 'scragvae')."
                        ),
                    },
                ),
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Target device for VAE decoding. 'cuda' is fast (~0.4-1.5 GB VRAM "
                            "with tiled decode); 'cpu' guarantees zero VRAM usage."
                        ),
                    },
                ),
                "precision": (
                    ["fp16", "bf16", "fp32"],
                    {
                        "default": "fp16",
                        "tooltip": (
                            "VAE precision on GPU ('fp16' matches upstream ACE-Step defaults; "
                            "'cpu' always uses float32)."
                        ),
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Automatically download missing official VAE weights from Hugging Face "
                            "into ComfyUI/models/acestep/."
                        ),
                    },
                ),
            }
        }

    def load_vae(
        self,
        model_path: str,
        device: str,
        precision: str,
        auto_download: bool,
    ) -> Tuple[AceStepVAEHandle]:
        models_dir = get_acestep_models_dir()
        vae_dir = ensure_component_available(model_path, category="vae", auto_download=auto_download)

        handle = AceStepVAEHandle(
            model_name=model_path,
            checkpoint_dir=models_dir,
            resolved_model_dir=vae_dir,
            device_mode=device,
            precision=precision,
            auto_download=auto_download,
            sample_rate=48000,
        )
        return (handle,)


NODE_CLASS_MAPPINGS = {
    "AceStepLMLoader": AceStepLMLoader,
    "AceStepDiTLoader": AceStepDiTLoader,
    "AceStepVAELoader": AceStepVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLMLoader": "ACE-Step 1.5 LM Loader (5Hz)",
    "AceStepDiTLoader": "ACE-Step 1.5 DiT Loader (2B / XL 4B)",
    "AceStepVAELoader": "ACE-Step 1.5 VAE Loader",
}
