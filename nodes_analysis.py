"""Audio Analysis, Understanding, LRC Generation, and Quality Scoring nodes for ComfyUI-Bao-Ace_Step_1_5.

All nodes strictly isolate DiT and 5Hz LM execution so a 12 GB VRAM GPU can
tokenize audio, reverse-engineer metadata/lyrics (Audio Understanding), align
LRC timestamps (cross-attention + DTW), and compute combined DiT + LM PMI
quality scores without OOM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

from .memory_utils import (
    AceStepDiTHandle,
    AceStepLMHandle,
    get_compute_device,
    move_module_to_device_safe,
    purge_vram,
    resolve_torch_dtype,
)
from .nodes_lm import (
    _coerce_optional_duration,
    _coerce_optional_int,
    _normalize_timesig,
)

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")


def _extract_codes_with_dit_handler(handler: Any, latents_btc: torch.Tensor) -> Tuple[str, int]:
    """Run ACE-Step DiT FSQ tokenizer on ``[1, T_25hz, 64]`` latents to produce 5Hz semantic codes."""
    exec_device = torch.device(handler.device)
    compute_dtype = handler.dtype

    if latents_btc.dim() == 2:
        latents_btc = latents_btc.unsqueeze(0)
    if latents_btc.shape[1] == 64 and latents_btc.shape[2] != 64:
        latents_btc = latents_btc.transpose(1, 2).contiguous()

    hidden_states = latents_btc[:1].to(device=exec_device, dtype=compute_dtype)
    attention_mask = torch.ones((1, hidden_states.shape[1]), dtype=torch.bool, device=exec_device)

    with torch.inference_mode():
        with handler._load_model_context("model"):
            silence = handler.silence_latent.to(device=exec_device, dtype=compute_dtype)
            _, indices, _ = handler.model.tokenize(hidden_states, silence, attention_mask)
            indices_flat = indices.flatten().detach().cpu().tolist()

    codes_string = "".join(f"<|audio_code_{int(idx)}|>" for idx in indices_flat)
    return codes_string, len(indices_flat)


def _save_lrc_to_comfy_output(lrc_text: str, filename_prefix: str) -> str:
    """Write an ``.lrc`` file into ComfyUI's ``output/`` directory using sequential numbering."""
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        out_dir = Path(folder_paths.get_output_directory())
    else:
        out_dir = Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_prefix = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in (filename_prefix or "acestep_lyrics"))
    safe_prefix = safe_prefix.strip("._") or "acestep_lyrics"

    counter = 1
    while True:
        candidate = out_dir / f"{safe_prefix}_{counter:04d}.lrc"
        if not candidate.exists():
            candidate.write_text(lrc_text, encoding="utf-8")
            return str(candidate)
        counter += 1


class AceStepAudioTokenizer:
    """Convert pre-encoded 25Hz audio latents into 5Hz semantic audio codes using the DiT FSQ tokenizer."""

    CATEGORY = "ACE-Step 1.5/Analysis"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("audio_codes", "token_count")
    FUNCTION = "tokenize_audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_model": (
                    "ACESTEP_DIT",
                    {"tooltip": "DiT handle from ACE-Step 1.5 DiT Loader (provides the FSQ audio tokenizer)."},
                ),
                "audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded 25Hz audio latent from ACE-Step 1.5 VAE Encode."},
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Mandatory for 12 GB VRAM: unload DiT immediately after tokenization "
                            "before passing audio codes to the 5Hz LM."
                        ),
                    },
                ),
            }
        }

    def tokenize_audio(
        self,
        dit_model: AceStepDiTHandle,
        audio_latent: Dict[str, Any],
        unload_after_run: bool,
    ) -> Tuple[str, int]:
        if not isinstance(audio_latent, dict) or "latents" not in audio_latent:
            raise ValueError("Invalid ACESTEP_AUDIO_LATENT input: connect ACE-Step 1.5 VAE Encode.")

        latents = audio_latent["latents"]
        try:
            handler = dit_model.ensure_loaded()
            codes_str, token_count = _extract_codes_with_dit_handler(handler, latents)
            logger.info("[ACE-Step 1.5] Tokenized audio latent into %d 5Hz semantic tokens.", token_count)
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading DiT after audio tokenization (unload_after_run=True).")
                dit_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        return (codes_str, int(token_count))


class AceStepAudioUnderstanding:
    """Analyze 5Hz semantic audio codes with the 5Hz LM to extract caption, lyrics, and metadata."""

    CATEGORY = "ACE-Step 1.5/Analysis"
    RETURN_TYPES = ("STRING", "STRING", "ACESTEP_METADATA", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "metadata", "understanding_json")
    FUNCTION = "understand_audio"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lm_model": (
                    "ACESTEP_LM",
                    {"tooltip": "5Hz LM handle from ACE-Step 1.5 LM Loader."},
                ),
                "audio_codes": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Serialized 5Hz semantic audio codes from ACE-Step 1.5 Audio Tokenizer.",
                    },
                ),
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature (lower values 0.2-0.5 yield more accurate transcription/metadata).",
                    },
                ),
                "lm_top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Top-K sampling cutoff (0 disables Top-K).",
                    },
                ),
                "lm_top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Nucleus sampling Top-P threshold.",
                    },
                ),
                "lm_repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Repetition penalty (1.0 disables penalty).",
                    },
                ),
                "use_constrained_decoding": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enforce structured metadata decoding with FSM constraints.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload 5Hz LM and purge CUDA/ComfyUI caches immediately after analysis.",
                    },
                ),
            }
        }

    def understand_audio(
        self,
        lm_model: AceStepLMHandle,
        audio_codes: str,
        lm_temperature: float,
        lm_top_k: int,
        lm_top_p: float,
        lm_repetition_penalty: float,
        use_constrained_decoding: bool,
        unload_after_run: bool,
    ) -> Tuple[str, str, Dict[str, Any], str]:
        codes = (audio_codes or "").strip()
        if not codes:
            raise ValueError(
                "AceStepAudioUnderstanding requires non-empty 5Hz audio_codes (connect ACE-Step 1.5 Audio Tokenizer)."
            )

        top_k_arg = lm_top_k if lm_top_k > 0 else None
        top_p_arg = lm_top_p if 0.0 < lm_top_p < 1.0 else None

        try:
            llm_handler = lm_model.ensure_loaded()
            logger.info("[ACE-Step 1.5] Running 5Hz LM Audio Understanding...")
            raw_meta, status_msg = llm_handler.understand_audio_from_codes(
                audio_codes=codes,
                temperature=lm_temperature,
                top_k=top_k_arg,
                top_p=top_p_arg,
                repetition_penalty=lm_repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
            )
            logger.info("[ACE-Step 1.5] Audio Understanding status: %s", status_msg)
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading 5Hz LM after Audio Understanding (unload_after_run=True).")
                lm_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        meta_dict = raw_meta if isinstance(raw_meta, dict) else {}
        caption = str(meta_dict.get("caption", "")).strip()
        lyrics = str(meta_dict.get("lyrics", "")).strip()
        bpm = _coerce_optional_int(meta_dict.get("bpm"), 30, 300)
        keyscale = str(meta_dict.get("keyscale", "")).strip()
        timesig = _normalize_timesig(meta_dict.get("timesignature"))
        duration = _coerce_optional_duration(meta_dict.get("duration"))
        language = str(meta_dict.get("language", "unknown")).strip() or "unknown"
        is_instrumental = lyrics.lower() in ("", "[instrumental]", "[inst]")

        structured_meta: Dict[str, Any] = {
            "caption": caption,
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesig,
            "duration": duration,
            "vocal_language": language,
            "instrumental": is_instrumental,
        }
        report = {
            **structured_meta,
            "status": status_msg,
        }
        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        return {
            "ui": {"text": [report_json]},
            "result": (caption, lyrics, structured_meta, report_json),
        }


class AceStepLRCGenerator:
    """Generate synchronized .lrc lyric timestamps via DiT cross-attention alignment and DTW."""

    CATEGORY = "ACE-Step 1.5/Analysis"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lrc_text", "timestamps_json", "saved_lrc_path")
    FUNCTION = "generate_lrc"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_model": (
                    "ACESTEP_DIT",
                    {"tooltip": "DiT handle from ACE-Step 1.5 DiT Loader."},
                ),
                "latent": (
                    "ACESTEP_LATENT",
                    {
                        "tooltip": (
                            "25Hz latent from ACE-Step 1.5 DiT Sampler with "
                            "'capture_alignment_data=True' enabled."
                        ),
                    },
                ),
                "inference_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Reference diffusion step count used to compute the alignment timestep t_last = 1/steps.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**32 - 1,
                        "tooltip": "Deterministic noise seed for cross-attention extraction.",
                    },
                ),
                "save_lrc_file": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Write the generated .lrc subtitle file into ComfyUI/output/.",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "acestep_lyrics",
                        "multiline": False,
                        "tooltip": "Filename prefix used when save_lrc_file=True.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload DiT and purge CUDA/ComfyUI caches immediately after LRC alignment.",
                    },
                ),
            }
        }

    def generate_lrc(
        self,
        dit_model: AceStepDiTHandle,
        latent: Dict[str, Any],
        inference_steps: int,
        seed: int,
        save_lrc_file: bool,
        filename_prefix: str,
        unload_after_run: bool,
    ) -> Tuple[str, str, str]:
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Invalid ACESTEP_LATENT input.")

        alignment = latent.get("alignment_data")
        if not isinstance(alignment, dict) or alignment.get("encoder_hidden_states") is None:
            raise ValueError(
                "ACESTEP_LATENT does not contain alignment tensors. Enable 'capture_alignment_data=True' "
                "on ACE-Step 1.5 DiT Sampler before running LRC Generation."
            )

        pred_latent = latent["samples"][:1]
        meta = latent.get("metadata", {})
        vocal_language = str(meta.get("vocal_language", "en") or "en")
        total_duration = float(latent.get("duration", pred_latent.shape[1] / 25.0))

        try:
            handler = dit_model.ensure_loaded()
            logger.info("[ACE-Step 1.5] Running DiT cross-attention LRC alignment (duration=%.2fs)...", total_duration)
            result = handler.get_lyric_timestamp(
                pred_latent=pred_latent,
                encoder_hidden_states=alignment["encoder_hidden_states"][:1],
                encoder_attention_mask=alignment["encoder_attention_mask"][:1],
                context_latents=alignment["context_latents"][:1],
                lyric_token_ids=alignment["lyric_token_ids"][:1],
                total_duration_seconds=total_duration,
                vocal_language=vocal_language,
                inference_steps=int(inference_steps),
                seed=int(seed),
            )
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading DiT after LRC generation (unload_after_run=True).")
                dit_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        if not result.get("success", False):
            raise RuntimeError(f"LRC timestamp generation failed: {result.get('error', 'Unknown error')}")

        lrc_text = str(result.get("lrc_text", "")).strip()
        serializable = {
            "success": True,
            "duration_seconds": total_duration,
            "vocal_language": vocal_language,
            "sentence_timestamps": result.get("sentence_timestamps", []),
            "token_timestamps": result.get("token_timestamps", []),
        }
        timestamps_json = json.dumps(serializable, indent=2, ensure_ascii=False, default=str)

        saved_path = ""
        if save_lrc_file and lrc_text:
            saved_path = _save_lrc_to_comfy_output(lrc_text, filename_prefix)
            logger.info("[ACE-Step 1.5] Saved LRC file to '%s'.", saved_path)

        return {
            "ui": {"text": [lrc_text or timestamps_json]},
            "result": (lrc_text, timestamps_json, saved_path),
        }


class AceStepQualityScorer:
    """Compute DiT lyric alignment scores and 5Hz LM PMI quality/reward scores sequentially in <= 12 GB VRAM.

    Execution order guarantees zero VRAM coexistence:
    1. DiT stage: evaluates ``get_lyric_score`` (and tokenizes 5Hz codes if needed), then UNLOADS DiT completely.
    2. LM stage: evaluates ``calculate_pmi_score_per_condition`` and ``calculate_reward_score``, then UNLOADS LM completely.
    """

    CATEGORY = "ACE-Step 1.5/Analysis"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("overall_score", "dit_lyric_score", "lm_alignment_score", "lm_pmi_reward", "report_json")
    FUNCTION = "score_generation"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": (
                    "ACESTEP_LATENT",
                    {
                        "tooltip": (
                            "25Hz latent payload from ACE-Step 1.5 DiT Sampler. Enable "
                            "'capture_alignment_data=True' on the sampler to compute DiT lyric alignment scores."
                        ),
                    },
                ),
                "run_dit_lyric_score": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compute DiT cross-attention lyric alignment scores (requires dit_model + capture_alignment_data).",
                    },
                ),
                "run_lm_pmi_score": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compute 5Hz LM Pointwise Mutual Information (PMI) & reward scores (requires lm_model).",
                    },
                ),
                "inference_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Reference diffusion steps for DiT alignment scoring.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**32 - 1,
                        "tooltip": "Deterministic noise seed for DiT alignment scoring.",
                    },
                ),
                "pmi_temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Temperature scaling for LM log-probability PMI evaluation.",
                    },
                ),
                "pmi_topk": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Top-K recall threshold used by LM PMI condition scoring.",
                    },
                ),
                "pmi_score_scale": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sigmoid sensitivity scale mapping raw PMI to [0, 1].",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Mandatory for 12 GB VRAM: strictly unload DiT before loading LM, and unload LM afterwards.",
                    },
                ),
            },
            "optional": {
                "dit_model": (
                    "ACESTEP_DIT",
                    {"tooltip": "Optional DiT handle for lyric alignment scoring and automatic 5Hz code tokenization."},
                ),
                "lm_model": (
                    "ACESTEP_LM",
                    {"tooltip": "Optional 5Hz LM handle for PMI quality & reward scoring."},
                ),
                "audio_codes": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional 5Hz semantic codes (if omitted, extracted automatically via dit_model or latent metadata).",
                    },
                ),
            },
        }

    def score_generation(
        self,
        latent: Dict[str, Any],
        run_dit_lyric_score: bool,
        run_lm_pmi_score: bool,
        inference_steps: int,
        seed: int,
        pmi_temperature: float,
        pmi_topk: int,
        pmi_score_scale: float,
        unload_after_run: bool,
        dit_model: Optional[AceStepDiTHandle] = None,
        lm_model: Optional[AceStepLMHandle] = None,
        audio_codes: str = "",
    ) -> Tuple[float, float, float, float, str]:
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Invalid ACESTEP_LATENT input.")

        pred_latent = latent["samples"][:1]
        meta = latent.get("metadata", {}) if isinstance(latent.get("metadata"), dict) else {}
        alignment = latent.get("alignment_data")

        resolved_codes = (audio_codes or "").strip() or str(latent.get("audio_codes", "") or "").strip()
        dit_lyric_score = 0.0
        lm_alignment_score = 0.0
        lm_pmi_reward = 0.0
        pmi_details: Dict[str, Any] = {}
        notes: List[str] = []

        # ------------------------------------------------------------------
        # Stage 1: DiT Alignment Score (+ optional 5Hz code extraction) -> UNLOAD DiT
        # ------------------------------------------------------------------
        needs_dit = (
            dit_model is not None
            and (
                (run_dit_lyric_score and isinstance(alignment, dict) and alignment.get("encoder_hidden_states") is not None)
                or (run_lm_pmi_score and not resolved_codes)
            )
        )
        if run_dit_lyric_score and (dit_model is None or not isinstance(alignment, dict)):
            notes.append(
                "DiT lyric alignment score skipped (requires dit_model and capture_alignment_data=True on AceStepSampler)."
            )

        if needs_dit and dit_model is not None:
            try:
                handler = dit_model.ensure_loaded()
                if run_dit_lyric_score and isinstance(alignment, dict) and alignment.get("encoder_hidden_states") is not None:
                    logger.info("[ACE-Step 1.5] Computing DiT lyric alignment score...")
                    score_res = handler.get_lyric_score(
                        pred_latent=pred_latent,
                        encoder_hidden_states=alignment["encoder_hidden_states"][:1],
                        encoder_attention_mask=alignment["encoder_attention_mask"][:1],
                        context_latents=alignment["context_latents"][:1],
                        lyric_token_ids=alignment["lyric_token_ids"][:1],
                        vocal_language=str(meta.get("vocal_language", "en") or "en"),
                        inference_steps=int(inference_steps),
                        seed=int(seed),
                    )
                    if score_res.get("success"):
                        dit_lyric_score = float(score_res.get("dit_score", 0.0))
                        lm_alignment_score = float(score_res.get("lm_score", 0.0))
                    else:
                        notes.append(f"DiT lyric score warning: {score_res.get('error', 'unknown')}")

                if run_lm_pmi_score and not resolved_codes:
                    logger.info("[ACE-Step 1.5] Extracting 5Hz codes from generated latent for LM PMI scoring...")
                    resolved_codes, _ = _extract_codes_with_dit_handler(handler, pred_latent)
            finally:
                # Always unload DiT before loading 5Hz LM so they never coexist in 12 GB VRAM!
                if unload_after_run or (run_lm_pmi_score and lm_model is not None):
                    logger.info("[ACE-Step 1.5] Unloading DiT before LM PMI stage.")
                    dit_model.unload(release_reference=True)
                else:
                    purge_vram(aggressive=False)

        # ------------------------------------------------------------------
        # Stage 2: 5Hz LM PMI Quality & Reward Scoring -> UNLOAD LM
        # ------------------------------------------------------------------
        if run_lm_pmi_score:
            if lm_model is None:
                notes.append("LM PMI scoring skipped (no lm_model connected).")
            elif not resolved_codes:
                notes.append("LM PMI scoring skipped (no 5Hz audio_codes available; connect dit_model or audio_codes).")
            else:
                from acestep.core.scoring.lm_score import (
                    calculate_pmi_score_per_condition,
                    calculate_reward_score,
                )

                try:
                    llm_handler = lm_model.ensure_loaded()
                    logger.info("[ACE-Step 1.5] Computing 5Hz LM PMI scores per condition...")
                    score_meta: Dict[str, Any] = {}
                    if meta.get("bpm") is not None:
                        score_meta["bpm"] = meta["bpm"]
                    if meta.get("keyscale"):
                        score_meta["keyscale"] = meta["keyscale"]
                    if meta.get("timesignature"):
                        score_meta["timesignature"] = meta["timesignature"]
                    if latent.get("duration"):
                        score_meta["duration"] = int(round(float(latent["duration"])))
                    if meta.get("vocal_language") and meta["vocal_language"] != "unknown":
                        score_meta["language"] = meta["vocal_language"]

                    pmi_scores, pmi_status = calculate_pmi_score_per_condition(
                        llm_handler=llm_handler,
                        audio_codes=resolved_codes,
                        caption=str(meta.get("caption", "")),
                        lyrics=str(meta.get("lyrics", "")),
                        metadata=score_meta if score_meta else None,
                        temperature=float(pmi_temperature),
                        topk=int(pmi_topk),
                        score_scale=float(pmi_score_scale),
                    )
                    pmi_details = dict(pmi_scores) if isinstance(pmi_scores, dict) else {}
                    pmi_details["status"] = pmi_status
                    if pmi_scores:
                        reward_res = calculate_reward_score(pmi_scores)
                        if isinstance(reward_res, dict):
                            lm_pmi_reward = float(
                                reward_res.get("final_score", reward_res.get("reward_score", 0.0))
                            )
                            pmi_details["reward_breakdown"] = reward_res
                        elif isinstance(reward_res, (int, float)):
                            lm_pmi_reward = float(reward_res)
                finally:
                    if unload_after_run:
                        logger.info("[ACE-Step 1.5] Unloading 5Hz LM after PMI scoring (unload_after_run=True).")
                        lm_model.unload(release_reference=True)
                    else:
                        purge_vram(aggressive=False)

        # Aggregate overall score cleanly across enabled components
        active_scores = [s for s in (dit_lyric_score, lm_alignment_score, lm_pmi_reward) if s > 0.0]
        overall_score = float(sum(active_scores) / len(active_scores)) if active_scores else 0.0

        report = {
            "overall_score": round(overall_score, 4),
            "dit_lyric_score": round(dit_lyric_score, 4),
            "lm_alignment_score": round(lm_alignment_score, 4),
            "lm_pmi_reward": round(lm_pmi_reward, 4),
            "pmi_details": pmi_details,
            "notes": notes,
        }
        report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        return {
            "ui": {"text": [report_json]},
            "result": (
                overall_score,
                dit_lyric_score,
                lm_alignment_score,
                lm_pmi_reward,
                report_json,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "AceStepAudioTokenizer": AceStepAudioTokenizer,
    "AceStepAudioUnderstanding": AceStepAudioUnderstanding,
    "AceStepLRCGenerator": AceStepLRCGenerator,
    "AceStepQualityScorer": AceStepQualityScorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioTokenizer": "ACE-Step 1.5 Audio Tokenizer (5Hz Codes)",
    "AceStepAudioUnderstanding": "ACE-Step 1.5 Audio Understanding (5Hz LM)",
    "AceStepLRCGenerator": "ACE-Step 1.5 LRC Generator",
    "AceStepQualityScorer": "ACE-Step 1.5 Quality Scorer (DiT + LM PMI)",
}
