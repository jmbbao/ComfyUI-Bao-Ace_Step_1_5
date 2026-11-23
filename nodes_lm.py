"""5Hz Language Model Planner node for ComfyUI-Bao-Ace_Step_1_5.

Supports:
- Simple Mode (Query Rewriting): expands a short natural-language description
  into rich musical tags/caption, structured lyrics ([Verse], [Chorus], ...),
  metadata (BPM, keyscale, time signature, duration, language), and optional
  5Hz semantic audio codes.
- Custom Mode (CoT + 5Hz Codes or Format + 5Hz Codes): enhances user-provided
  prompts/lyrics and generates 5Hz semantic planning tokens.
- Multilingual vocals across all 50+ ACE-Step 1.5 supported languages and
  instrumental-only generation.
- Strict VRAM cleanup via ``unload_after_run=True`` so the 4B LM is completely
  purged from GPU memory before the DiT stage begins.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from comfy.utils import ProgressBar
except ImportError:  # pragma: no cover
    ProgressBar = None

from .memory_utils import AceStepLMHandle, purge_vram

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")

# Aligned with acestep.constants (kept locally so ComfyUI can inspect INPUT_TYPES
# cleanly even before the first runtime import of acestep).
VALID_LANGUAGES: List[str] = [
    "unknown", "en", "es", "zh", "ja", "ko", "fr", "de", "it", "pt", "ru",
    "ar", "az", "bg", "bn", "ca", "cs", "da", "el", "fa", "fi", "he", "hi",
    "hr", "ht", "hu", "id", "is", "la", "lt", "ms", "ne", "nl", "no", "pa",
    "pl", "ro", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr",
    "uk", "ur", "vi", "yue",
]

KEYSCALE_OPTIONS: List[str] = ["auto"] + [
    f"{note}{acc} {mode}"
    for note in ("C", "D", "E", "F", "G", "A", "B")
    for acc in ("", "#", "b")
    for mode in ("major", "minor")
]

TIMESIGNATURE_OPTIONS: List[str] = ["auto", "4", "3", "2", "6"]

PLANNER_MODES: List[str] = [
    "simple_query_rewrite",
    "custom_cot_and_codes",
    "custom_format_and_codes",
    "bypass",
]


def _coerce_optional_int(value: Any, min_val: int, max_val: int) -> Optional[int]:
    if value is None or value == "" or value == "auto" or value == "N/A":
        return None
    try:
        parsed = int(float(value))
        if min_val <= parsed <= max_val:
            return parsed
    except (TypeError, ValueError):
        pass
    return None


def _coerce_optional_duration(value: Any) -> Optional[float]:
    if value is None or value == "" or value == "auto" or value == "N/A":
        return None
    try:
        parsed = float(value)
        if parsed > 0:
            return max(10.0, min(600.0, parsed))
    except (TypeError, ValueError):
        pass
    return None


def _normalize_timesig(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in ("", "auto", "N/A", "none", "None"):
        return ""
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    return text if text in ("2", "3", "4", "6") else ""


def _resolve_seed_list(seed: int, batch_size: int) -> List[int]:
    base = random.randint(0, 2**32 - 1) if seed < 0 else int(seed)
    return [(base + i) % (2**32) for i in range(batch_size)]


class AceStepLMPlanner:
    """Plan song metadata, structured lyrics, and 5Hz semantic tokens using ACE-Step's 5Hz LM."""

    CATEGORY = "ACE-Step 1.5/LM"
    RETURN_TYPES = ("ACESTEP_PLAN", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lm_plan", "caption", "lyrics", "metadata_json", "audio_codes")
    FUNCTION = "plan_music"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lm_model": (
                    "ACESTEP_LM",
                    {"tooltip": "5Hz LM handle from ACE-Step 1.5 LM Loader."},
                ),
                "mode": (
                    PLANNER_MODES,
                    {
                        "default": "simple_query_rewrite",
                        "tooltip": (
                            "'simple_query_rewrite': expand a short natural-language idea into "
                            "caption, lyrics, metadata, and 5Hz codes. "
                            "'custom_cot_and_codes': keep your prompt/lyrics and run CoT + 5Hz codes. "
                            "'custom_format_and_codes': rewrite/format your caption & metadata before codes. "
                            "'bypass': skip LM execution and pack inputs directly into ACESTEP_PLAN."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "warm indie folk ballad with fingerpicked acoustic guitar, soft cello, and intimate vocals",
                        "multiline": True,
                        "tooltip": (
                            "In Simple Mode, write a natural-language song idea. In Custom Mode, "
                            "write detailed style tags and musical caption."
                        ),
                    },
                ),
                "lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional structured lyrics using tags such as [Verse 1], [Pre-Chorus], "
                            "[Chorus], [Bridge], [Outro]. Leave blank in Simple Mode for auto-lyrics, "
                            "or enable 'instrumental' for non-vocal tracks."
                        ),
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Generate purely instrumental music without vocals.",
                    },
                ),
                "vocal_language": (
                    VALID_LANGUAGES,
                    {
                        "default": "en",
                        "tooltip": "Vocal language code (50+ languages supported; 'unknown' lets the LM decide).",
                    },
                ),
                "bpm": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "step": 1,
                        "tooltip": "Tempo in BPM (30-300). Set to 0 to let the LM infer BPM automatically.",
                    },
                ),
                "keyscale": (
                    KEYSCALE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Musical key and mode (e.g. 'C major', 'F# minor'). 'auto' lets the LM choose.",
                    },
                ),
                "timesignature": (
                    TIMESIGNATURE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Musical time signature numerator (2=2/4, 3=3/4, 4=4/4, 6=6/8) or 'auto'.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": (
                            "Target song duration in seconds (10s to 600s). "
                            "Set to -1 to let the 5Hz LM decide duration automatically."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of variations to plan (keep at 1 for 12 GB VRAM with 4B models).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": -1,
                        "max": 2**32 - 1,
                        "tooltip": "Random seed for LM planning (-1 for random).",
                    },
                ),
                "generate_audio_codes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Generate 5Hz semantic audio tokens to guide musical structure, "
                            "melody, and vocal timing in the DiT sampler."
                        ),
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Mandatory for 12 GB VRAM: unload the 5Hz LM and purge CUDA/ComfyUI "
                            "caches immediately after planning so DiT XL has clean VRAM."
                        ),
                    },
                ),
            },
            "optional": {
                "use_cot_metas": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow the LM to fill missing metadata fields via Chain-of-Thought.",
                    },
                ),
                "use_cot_caption": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow the LM to expand/refine the musical caption via Chain-of-Thought.",
                    },
                ),
                "use_cot_language": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow the LM to detect or refine the vocal language via Chain-of-Thought.",
                    },
                ),
                "use_constrained_decoding": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use FSM-constrained decoding to enforce valid metadata ranges and duration limits.",
                    },
                ),
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature for the 5Hz LM.",
                    },
                ),
                "lm_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Classifier-Free Guidance scale during 5Hz semantic token generation.",
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
                        "tooltip": "Nucleus sampling threshold Top-P.",
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
                "lm_negative_prompt": (
                    "STRING",
                    {
                        "default": "NO USER INPUT",
                        "multiline": False,
                        "tooltip": "Negative prompt used when lm_cfg_scale > 1.0.",
                    },
                ),
                "metadata": (
                    "ACESTEP_METADATA",
                    {
                        "tooltip": (
                            "Optional structured metadata from ACE-Step 1.5 Metadata Control, "
                            "Query Rewriter, or Audio Understanding. Overrides empty or auto fields."
                        ),
                    },
                ),
            },
        }

    def plan_music(
        self,
        lm_model: AceStepLMHandle,
        mode: str,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        vocal_language: str,
        bpm: int,
        keyscale: str,
        timesignature: str,
        duration: float,
        batch_size: int,
        seed: int,
        generate_audio_codes: bool,
        unload_after_run: bool,
        use_cot_metas: bool = True,
        use_cot_caption: bool = True,
        use_cot_language: bool = True,
        use_constrained_decoding: bool = True,
        lm_temperature: float = 0.85,
        lm_cfg_scale: float = 2.0,
        lm_top_k: int = 0,
        lm_top_p: float = 0.9,
        lm_repetition_penalty: float = 1.0,
        lm_negative_prompt: str = "NO USER INPUT",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, str, str, str]:
        pbar = ProgressBar(3) if ProgressBar is not None else None
        seed_list = _resolve_seed_list(seed, batch_size)
        meta_in = metadata if isinstance(metadata, dict) else {}

        is_instrumental = instrumental or bool(meta_in.get("instrumental", False))
        resolved_caption = str(meta_in.get("caption", "")).strip() or (prompt or "").strip()
        if is_instrumental:
            resolved_lyrics = "[Instrumental]"
        else:
            resolved_lyrics = (lyrics or "").strip() or str(meta_in.get("lyrics", "")).strip()

        resolved_bpm = (
            _coerce_optional_int(bpm, 30, 300)
            if bpm > 0
            else _coerce_optional_int(meta_in.get("bpm"), 30, 300)
        )
        resolved_keyscale = (
            str(meta_in.get("keyscale", "")).strip()
            if keyscale == "auto"
            else keyscale
        )
        resolved_timesig = (
            _normalize_timesig(meta_in.get("timesignature"))
            if timesignature == "auto"
            else _normalize_timesig(timesignature)
        )
        resolved_duration = (
            _coerce_optional_duration(meta_in.get("duration"))
            if (duration <= 0 and meta_in.get("duration") is not None)
            else _coerce_optional_duration(duration)
        )
        if is_instrumental:
            resolved_language = "unknown"
        elif vocal_language and vocal_language != "unknown":
            resolved_language = vocal_language
        else:
            resolved_language = str(meta_in.get("vocal_language", "unknown") or "unknown")
        instrumental = is_instrumental

        user_metadata: Dict[str, Any] = {}
        if resolved_bpm is not None:
            user_metadata["bpm"] = resolved_bpm
        if resolved_keyscale:
            user_metadata["keyscale"] = resolved_keyscale
        if resolved_timesig:
            user_metadata["timesignature"] = resolved_timesig
        if resolved_duration is not None:
            user_metadata["duration"] = int(round(resolved_duration))
        if resolved_language and resolved_language != "unknown":
            user_metadata["language"] = resolved_language

        audio_codes_list: List[str] = [""] * batch_size
        cot_metadata_list: List[Dict[str, Any]] = []

        if mode == "bypass":
            plan = self._build_plan_payload(
                caption=resolved_caption,
                lyrics=resolved_lyrics,
                instrumental=instrumental,
                vocal_language=resolved_language,
                bpm=resolved_bpm,
                keyscale=resolved_keyscale,
                timesignature=resolved_timesig,
                duration=resolved_duration if resolved_duration is not None else 60.0,
                audio_codes=audio_codes_list,
                batch_size=batch_size,
                seeds=seed_list,
                mode=mode,
                lm_model_name=lm_model.model_name,
            )
            meta_json = json.dumps(plan["metadata"], indent=2, ensure_ascii=False)
            return (plan, resolved_caption, resolved_lyrics, meta_json, "")

        top_k_arg = lm_top_k if lm_top_k > 0 else None
        top_p_arg = lm_top_p if 0.0 < lm_top_p < 1.0 else None

        try:
            llm_handler = lm_model.ensure_loaded()
            if pbar is not None:
                pbar.update(1)

            # Stage 1: Query Rewriting (Simple Mode) or Prompt Formatting (Custom Format Mode)
            if mode == "simple_query_rewrite":
                logger.info("[ACE-Step 1.5] Running 5Hz LM Simple Mode query expansion...")
                sample_meta, status_msg = llm_handler.create_sample_from_query(
                    query=resolved_caption,
                    instrumental=instrumental,
                    vocal_language=None if resolved_language == "unknown" else resolved_language,
                    temperature=lm_temperature,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    repetition_penalty=lm_repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                )
                logger.info("[ACE-Step 1.5] Simple Mode status: %s", status_msg)
                if isinstance(sample_meta, dict) and sample_meta:
                    if sample_meta.get("caption"):
                        resolved_caption = str(sample_meta["caption"]).strip()
                    if not instrumental and not resolved_lyrics and sample_meta.get("lyrics"):
                        resolved_lyrics = str(sample_meta["lyrics"]).strip()
                    if resolved_bpm is None:
                        resolved_bpm = _coerce_optional_int(sample_meta.get("bpm"), 30, 300)
                    if not resolved_keyscale and sample_meta.get("keyscale"):
                        resolved_keyscale = str(sample_meta["keyscale"]).strip()
                    if not resolved_timesig and sample_meta.get("timesignature"):
                        resolved_timesig = _normalize_timesig(sample_meta.get("timesignature"))
                    if resolved_duration is None:
                        resolved_duration = _coerce_optional_duration(sample_meta.get("duration"))
                    if not instrumental and resolved_language == "unknown" and sample_meta.get("language"):
                        resolved_language = str(sample_meta["language"]).strip()

            elif mode == "custom_format_and_codes":
                logger.info("[ACE-Step 1.5] Running 5Hz LM Custom Format & Metadata enhancement...")
                fmt_meta, status_msg = llm_handler.format_sample_from_input(
                    caption=resolved_caption,
                    lyrics=resolved_lyrics,
                    user_metadata=user_metadata if user_metadata else None,
                    temperature=lm_temperature,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    repetition_penalty=lm_repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                )
                logger.info("[ACE-Step 1.5] Format Mode status: %s", status_msg)
                if isinstance(fmt_meta, dict) and fmt_meta:
                    if fmt_meta.get("caption"):
                        resolved_caption = str(fmt_meta["caption"]).strip()
                    if not instrumental and fmt_meta.get("lyrics"):
                        resolved_lyrics = str(fmt_meta["lyrics"]).strip()
                    if resolved_bpm is None:
                        resolved_bpm = _coerce_optional_int(fmt_meta.get("bpm"), 30, 300)
                    if not resolved_keyscale and fmt_meta.get("keyscale"):
                        resolved_keyscale = str(fmt_meta["keyscale"]).strip()
                    if not resolved_timesig and fmt_meta.get("timesignature"):
                        resolved_timesig = _normalize_timesig(fmt_meta.get("timesignature"))
                    if resolved_duration is None:
                        resolved_duration = _coerce_optional_duration(fmt_meta.get("duration"))
                    if not instrumental and resolved_language == "unknown" and fmt_meta.get("language"):
                        resolved_language = str(fmt_meta["language"]).strip()

            if pbar is not None:
                pbar.update(1)

            # Refresh user_metadata after Simple/Format stage so CoT/5Hz codes stay consistent
            cot_user_metadata: Dict[str, Any] = {}
            if resolved_bpm is not None:
                cot_user_metadata["bpm"] = resolved_bpm
            if resolved_keyscale:
                cot_user_metadata["keyscale"] = resolved_keyscale
            if resolved_timesig:
                cot_user_metadata["timesignature"] = resolved_timesig
            if resolved_duration is not None:
                cot_user_metadata["duration"] = int(round(resolved_duration))
            if resolved_language and resolved_language != "unknown":
                cot_user_metadata["language"] = resolved_language

            # Stage 2: Chain-of-Thought + 5Hz Semantic Audio Codes
            run_cot_stage = generate_audio_codes or (mode == "custom_cot_and_codes")
            if run_cot_stage:
                infer_type = "llm_dit" if generate_audio_codes else "cot_only"
                logger.info(
                    "[ACE-Step 1.5] Running 5Hz LM generate_with_stop_condition (infer_type=%s, batch=%d)...",
                    infer_type,
                    batch_size,
                )
                lm_result = llm_handler.generate_with_stop_condition(
                    caption=resolved_caption,
                    lyrics=resolved_lyrics,
                    infer_type=infer_type,
                    temperature=lm_temperature,
                    cfg_scale=lm_cfg_scale,
                    negative_prompt=lm_negative_prompt,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    repetition_penalty=lm_repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    target_duration=resolved_duration,
                    user_metadata=cot_user_metadata if cot_user_metadata else None,
                    use_cot_metas=use_cot_metas,
                    use_cot_caption=use_cot_caption and (mode == "custom_cot_and_codes"),
                    use_cot_language=use_cot_language and not instrumental,
                    batch_size=batch_size,
                    seeds=seed_list,
                )

                if not lm_result.get("success", True):
                    err = lm_result.get("error", "Unknown 5Hz LM generation error")
                    raise RuntimeError(f"ACE-Step 5Hz LM planning failed: {err}")

                raw_codes = lm_result.get("audio_codes", "")
                if isinstance(raw_codes, list):
                    audio_codes_list = [str(c or "") for c in raw_codes]
                else:
                    audio_codes_list = [str(raw_codes or "")] * batch_size

                raw_meta = lm_result.get("metadata", {})
                if isinstance(raw_meta, list):
                    cot_metadata_list = [m for m in raw_meta if isinstance(m, dict)]
                elif isinstance(raw_meta, dict):
                    cot_metadata_list = [raw_meta]

                if cot_metadata_list:
                    first_meta = cot_metadata_list[0]
                    if use_cot_caption and mode == "custom_cot_and_codes" and first_meta.get("caption"):
                        resolved_caption = str(first_meta["caption"]).strip()
                    if resolved_bpm is None:
                        resolved_bpm = _coerce_optional_int(first_meta.get("bpm"), 30, 300)
                    if not resolved_keyscale and first_meta.get("keyscale"):
                        resolved_keyscale = str(first_meta["keyscale"]).strip()
                    if not resolved_timesig and first_meta.get("timesignature"):
                        resolved_timesig = _normalize_timesig(first_meta.get("timesignature"))
                    if resolved_duration is None:
                        resolved_duration = _coerce_optional_duration(first_meta.get("duration"))
                    if not instrumental and resolved_language == "unknown" and first_meta.get("language"):
                        resolved_language = str(first_meta["language"]).strip()

            if pbar is not None:
                pbar.update(1)
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading 5Hz LM after planning (unload_after_run=True).")
                lm_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        final_duration = resolved_duration if resolved_duration is not None else 60.0
        plan = self._build_plan_payload(
            caption=resolved_caption,
            lyrics=resolved_lyrics,
            instrumental=instrumental,
            vocal_language=resolved_language,
            bpm=resolved_bpm,
            keyscale=resolved_keyscale,
            timesignature=resolved_timesig,
            duration=final_duration,
            audio_codes=audio_codes_list,
            batch_size=batch_size,
            seeds=seed_list,
            mode=mode,
            lm_model_name=lm_model.model_name,
        )

        meta_json = json.dumps(plan["metadata"], indent=2, ensure_ascii=False)
        codes_preview = audio_codes_list[0] if audio_codes_list else ""
        return (plan, resolved_caption, resolved_lyrics, meta_json, codes_preview)

    @staticmethod
    def _build_plan_payload(
        *,
        caption: str,
        lyrics: str,
        instrumental: bool,
        vocal_language: str,
        bpm: Optional[int],
        keyscale: str,
        timesignature: str,
        duration: float,
        audio_codes: List[str],
        batch_size: int,
        seeds: List[int],
        mode: str,
        lm_model_name: str,
    ) -> Dict[str, Any]:
        """Build a CPU-resident, Phase-2-extensible ``ACESTEP_PLAN`` dictionary."""
        return {
            "version": 1,
            "task_type": "text2music",
            "mode": mode,
            "lm_model": lm_model_name,
            "caption": caption,
            "lyrics": lyrics,
            "instrumental": instrumental,
            "audio_codes": audio_codes,
            "batch_size": batch_size,
            "seeds": seeds,
            "metadata": {
                "bpm": bpm,
                "keyscale": keyscale,
                "timesignature": timesignature,
                "duration": float(duration),
                "vocal_language": vocal_language,
                "instrumental": instrumental,
            },
            # Reserved optional slots for Phase 2 (Reference Audio, Cover, Repaint, Multi-Track)
            "reference_audio": None,
            "src_audio": None,
            "repaint_window": None,
        }


NODE_CLASS_MAPPINGS = {
    "AceStepLMPlanner": AceStepLMPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLMPlanner": "ACE-Step 1.5 LM Planner (5Hz)",
}
