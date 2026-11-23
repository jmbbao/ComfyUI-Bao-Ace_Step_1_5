"""Dedicated task conditioning, metadata control, and query rewriting nodes for ComfyUI-Bao-Ace_Step_1_5.

Covers Phase 2 production tasks while maintaining strict 12 GB VRAM isolation:
- ``AceStepReferenceAudio``: Style/timbre reference conditioning from pre-encoded VAE latents.
- ``AceStepCoverTask``: Cover & Remix generation (``cover``, ``cover-nofsq``, and Flow-Edit morph overlay).
- ``AceStepRepaintTask``: Temporal inpainting/editing with mode/strength step-injection & waveform splice parameters.
- ``AceStepTrackExtractTask``: Stem/track separation (``extract``, base/xl-base DiT).
- ``AceStepMultiTrackTask``: Multi-track layer generation (``lego`` / ``complete``, base/xl-base DiT).
- ``AceStepVocal2BGMTask``: Dedicated vocal-to-full-accompaniment builder (``complete`` over vocal stem).
- ``AceStepMetadataControl``: Granular musical metadata & tag composer (``ACESTEP_METADATA``).
- ``AceStepQueryRewriter``: Advanced 5Hz LM query rewriting & structured prompt enhancement.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .memory_utils import AceStepLMHandle, purge_vram
from .nodes_lm import (
    KEYSCALE_OPTIONS,
    TIMESIGNATURE_OPTIONS,
    VALID_LANGUAGES,
    _coerce_optional_duration,
    _coerce_optional_int,
    _normalize_timesig,
)

logger = logging.getLogger("ComfyUI-Bao-Ace_Step_1_5")

# Official ACE-Step 1.5 track classes (aligned with acestep.constants.TRACK_NAMES)
TRACK_NAMES: List[str] = [
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
]

KEY_NOTES: List[str] = [
    "auto",
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
]

KEY_MODES: List[str] = ["major", "minor"]


def _resolve_repaint_config(mode: str = "balanced", strength: float = 0.5) -> Tuple[float, int, float]:
    """Mirror ``acestep.core.generation.handler.generate_music._resolve_repaint_config``.

    Returns:
        Tuple ``(injection_ratio, crossfade_frames, wav_crossfade_sec)``.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if mode == "aggressive":
        return 0.0, 0, 0.0
    if mode == "conservative":
        return 1.0, 25, 0.05
    inv = 1.0 - strength
    return inv, int(round(25 * inv)), float(0.05 * inv)


def _validate_audio_latent(audio_latent: Dict[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(audio_latent, dict) or "latents" not in audio_latent:
        raise ValueError(
            f"Invalid {label}: expected an ACESTEP_AUDIO_LATENT produced by 'ACE-Step 1.5 VAE Encode'."
        )
    if not isinstance(audio_latent["latents"], torch.Tensor):
        raise TypeError(f"{label}['latents'] must be a torch.Tensor.")
    return audio_latent


class AceStepReferenceAudio:
    """Prepare a pre-encoded 25Hz reference audio latent for timbre/style conditioning."""

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_REFERENCE_AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "build_reference"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Encoded audio latent from ACE-Step 1.5 VAE Encode."},
                ),
                "start_offset_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Start offset (seconds) into the encoded reference clip.",
                    },
                ),
                "max_duration_sec": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 2.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": (
                            "Reference excerpt window in seconds (ACE-Step defaults to 30 s = 750 frames at 25 Hz)."
                        ),
                    },
                ),
            }
        }

    def build_reference(
        self,
        audio_latent: Dict[str, Any],
        start_offset_sec: float,
        max_duration_sec: float,
    ) -> Tuple[Dict[str, Any]]:
        data = _validate_audio_latent(audio_latent, "audio_latent")
        latents = data["latents"].detach().cpu()  # [1, T, 64]
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)

        start_frame = max(0, int(round(start_offset_sec * 25.0)))
        max_frames = max(25, int(round(max_duration_sec * 25.0)))
        total_frames = latents.shape[1]
        if start_frame >= total_frames:
            start_frame = 0
        end_frame = min(total_frames, start_frame + max_frames)
        cropped = latents[:, start_frame:end_frame, :].contiguous()

        ref_payload: Dict[str, Any] = {
            "latents": cropped,  # [1, T_ref, 64] CPU
            "duration": cropped.shape[1] / 25.0,
            "start_offset_sec": float(start_offset_sec),
            "sample_rate": int(data.get("sample_rate", 48000)),
        }
        return (ref_payload,)


class AceStepCoverTask:
    """Configure Cover / Remix generation (including cover-nofsq and Flow-Edit morph overlay)."""

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_TASK_CONFIG",)
    RETURN_NAMES = ("task_config",)
    FUNCTION = "build_cover_task"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded source song latent from ACE-Step 1.5 VAE Encode."},
                ),
                "cover_mode": (
                    ["cover", "cover-nofsq"],
                    {
                        "default": "cover",
                        "tooltip": (
                            "'cover': standard semantic cover using FSQ audio structure tokens; "
                            "'cover-nofsq': continuous latent cover without FSQ quantization."
                        ),
                    },
                ),
                "audio_cover_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "How strongly the output adheres to the source song melody/structure "
                            "(1.0 = strict cover, 0.3-0.7 = creative remix)."
                        ),
                    },
                ),
                "cover_noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Optional noise blending strength toward the source audio latent "
                            "(0.0 = pure semantic cover; higher values keep closer acoustic texture)."
                        ),
                    },
                ),
                "enable_flow_edit_morph": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable Flow-Edit smooth morphing overlay (V_delta integration). "
                            "Requires a base or sft DiT checkpoint (base, sft, xl-base, xl-sft)."
                        ),
                    },
                ),
            },
            "optional": {
                "flow_edit_source_caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Caption describing the ORIGINAL source audio (used when enable_flow_edit_morph=True).",
                    },
                ),
                "flow_edit_source_lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Original source lyrics (used when enable_flow_edit_morph=True).",
                    },
                ),
                "flow_edit_n_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Flow-Edit minimum noise ratio bound (default 0.0).",
                    },
                ),
                "flow_edit_n_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Flow-Edit maximum noise ratio bound (default 1.0).",
                    },
                ),
                "flow_edit_n_avg": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of Monte-Carlo velocity averaging passes per Flow-Edit step.",
                    },
                ),
                "source_audio_codes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional pre-extracted 5Hz semantic codes from ACE-Step 1.5 Audio Tokenizer.",
                    },
                ),
            },
        }

    def build_cover_task(
        self,
        source_audio_latent: Dict[str, Any],
        cover_mode: str,
        audio_cover_strength: float,
        cover_noise_strength: float,
        enable_flow_edit_morph: bool,
        flow_edit_source_caption: str = "",
        flow_edit_source_lyrics: str = "",
        flow_edit_n_min: float = 0.0,
        flow_edit_n_max: float = 1.0,
        flow_edit_n_avg: int = 1,
        source_audio_codes: str = "",
    ) -> Tuple[Dict[str, Any]]:
        src = _validate_audio_latent(source_audio_latent, "source_audio_latent")
        config: Dict[str, Any] = {
            "task_type": cover_mode,
            "instruction": "Generate audio semantic tokens based on the given conditions:",
            "requires_base_model": False,
            "requires_flow_edit_support": bool(enable_flow_edit_morph),
            "source_audio_latent": src,
            "source_audio_codes": (source_audio_codes or "").strip(),
            "audio_cover_strength": float(audio_cover_strength),
            "cover_noise_strength": float(cover_noise_strength),
            "flow_edit_morph": bool(enable_flow_edit_morph),
            "flow_edit_source_caption": (flow_edit_source_caption or "").strip(),
            "flow_edit_source_lyrics": (flow_edit_source_lyrics or "").strip(),
            "flow_edit_n_min": float(flow_edit_n_min),
            "flow_edit_n_max": float(flow_edit_n_max),
            "flow_edit_n_avg": int(flow_edit_n_avg),
            "duration": float(src.get("duration", 60.0)),
        }
        return (config,)


class AceStepRepaintTask:
    """Configure temporal Repaint & Edit inpainting over a source audio track."""

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_TASK_CONFIG",)
    RETURN_NAMES = ("task_config",)
    FUNCTION = "build_repaint_task"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded source audio latent from ACE-Step 1.5 VAE Encode."},
                ),
                "repainting_start": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.25,
                        "tooltip": "Start time (seconds) of the region to regenerate/edit.",
                    },
                ),
                "repainting_end": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 0.25,
                        "tooltip": "End time (seconds) of the region to regenerate (-1 extends to track end).",
                    },
                ),
                "repaint_mode": (
                    ["balanced", "conservative", "aggressive"],
                    {
                        "default": "balanced",
                        "tooltip": (
                            "'conservative': maximum preservation of surrounding audio with wide crossfades; "
                            "'balanced': blends preservation and creative freedom via repaint_strength; "
                            "'aggressive': pure diffusion inside mask without step injection."
                        ),
                    },
                ),
                "repaint_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Edit strength used in 'balanced' mode (0.0 = max preservation, 1.0 = full regeneration).",
                    },
                ),
                "chunk_mask_mode": (
                    ["auto", "explicit"],
                    {
                        "default": "explicit",
                        "tooltip": "'explicit' restricts edits strictly to [repainting_start, repainting_end]; 'auto' lets the model adapt.",
                    },
                ),
                "enable_waveform_splice": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Preserve the bit-exact original waveform outside the repaint interval during "
                            "ACE-Step 1.5 VAE Decode with smooth boundary crossfading."
                        ),
                    },
                ),
            },
            "optional": {
                "wav_crossfade_sec": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Custom waveform splice crossfade duration in seconds (-1 = auto from repaint_mode/strength).",
                    },
                ),
            },
        }

    def build_repaint_task(
        self,
        source_audio_latent: Dict[str, Any],
        repainting_start: float,
        repainting_end: float,
        repaint_mode: str,
        repaint_strength: float,
        chunk_mask_mode: str,
        enable_waveform_splice: bool,
        wav_crossfade_sec: float = -1.0,
    ) -> Tuple[Dict[str, Any]]:
        src = _validate_audio_latent(source_audio_latent, "source_audio_latent")
        src_duration = float(src.get("duration", 60.0))
        resolved_end = src_duration if repainting_end < 0 else min(src_duration, float(repainting_end))
        resolved_start = max(0.0, min(float(repainting_start), max(0.0, resolved_end - 0.1)))

        injection_ratio, crossfade_frames, auto_wav_xfade = _resolve_repaint_config(
            mode=repaint_mode,
            strength=repaint_strength,
        )
        final_wav_xfade = float(wav_crossfade_sec) if wav_crossfade_sec >= 0.0 else auto_wav_xfade

        config: Dict[str, Any] = {
            "task_type": "repaint",
            "instruction": "Repaint the mask area based on the given conditions:",
            "requires_base_model": False,
            "requires_flow_edit_support": False,
            "source_audio_latent": src,
            "repainting_start": resolved_start,
            "repainting_end": resolved_end,
            "repaint_mode": repaint_mode,
            "repaint_strength": float(repaint_strength),
            "repaint_injection_ratio": float(injection_ratio),
            "repaint_crossfade_frames": int(crossfade_frames),
            "chunk_mask_mode": chunk_mask_mode,
            "enable_waveform_splice": bool(enable_waveform_splice),
            "wav_crossfade_sec": float(final_wav_xfade),
            "duration": src_duration,
        }
        return (config,)


class AceStepTrackExtractTask:
    """Configure Track Separation (Extract) to isolate a specific stem from mixed audio.

    Requires a pure ``base`` DiT variant (``acestep-v15-base`` or ``acestep-v15-xl-base``).
    """

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_TASK_CONFIG",)
    RETURN_NAMES = ("task_config",)
    FUNCTION = "build_extract_task"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded mixed audio latent from ACE-Step 1.5 VAE Encode."},
                ),
                "track_name": (
                    TRACK_NAMES,
                    {
                        "default": "vocals",
                        "tooltip": "Target stem to isolate from the source mix (e.g. vocals, drums, bass, guitar).",
                    },
                ),
            },
            "optional": {
                "region_start_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Optional start time (seconds) for selective extraction.",
                    },
                ),
                "region_end_sec": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Optional end time (seconds) (-1 extracts full duration).",
                    },
                ),
            },
        }

    def build_extract_task(
        self,
        source_audio_latent: Dict[str, Any],
        track_name: str,
        region_start_sec: float = 0.0,
        region_end_sec: float = -1.0,
    ) -> Tuple[Dict[str, Any]]:
        src = _validate_audio_latent(source_audio_latent, "source_audio_latent")
        src_duration = float(src.get("duration", 60.0))
        track_upper = track_name.strip().upper()
        instruction = f"Extract the {track_upper} track from the audio:"

        config: Dict[str, Any] = {
            "task_type": "extract",
            "instruction": instruction,
            "track_name": track_name,
            "requires_base_model": True,
            "requires_flow_edit_support": False,
            "source_audio_latent": src,
            "repainting_start": float(region_start_sec),
            "repainting_end": None if region_end_sec < 0 else min(src_duration, float(region_end_sec)),
            "duration": src_duration,
        }
        return (config,)


class AceStepMultiTrackTask:
    """Configure Multi-Track Generation (Lego layer addition or Complete arrangement).

    Requires a pure ``base`` DiT variant (``acestep-v15-base`` or ``acestep-v15-xl-base``).
    """

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_TASK_CONFIG",)
    RETURN_NAMES = ("task_config",)
    FUNCTION = "build_multitrack_task"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded backing or partial stem latent from ACE-Step 1.5 VAE Encode."},
                ),
                "operation": (
                    ["lego", "complete"],
                    {
                        "default": "lego",
                        "tooltip": (
                            "'lego': synthesize a single new instrument layer over the audio context; "
                            "'complete': arrange multiple complementary tracks around the input audio."
                        ),
                    },
                ),
                "primary_track": (
                    TRACK_NAMES,
                    {
                        "default": "guitar",
                        "tooltip": "Target track for 'lego' (or first target track class for 'complete').",
                    },
                ),
                "additional_tracks": (
                    "STRING",
                    {
                        "default": "drums, bass",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated additional track names used in 'complete' mode "
                            "(valid: woodwinds, brass, fx, synth, strings, percussion, keyboard, "
                            "guitar, bass, drums, backing_vocals, vocals)."
                        ),
                    },
                ),
            },
            "optional": {
                "region_start_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "Start time (seconds) for Lego region injection.",
                    },
                ),
                "region_end_sec": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "End time (seconds) for Lego region (-1 spans full track).",
                    },
                ),
            },
        }

    def build_multitrack_task(
        self,
        source_audio_latent: Dict[str, Any],
        operation: str,
        primary_track: str,
        additional_tracks: str,
        region_start_sec: float = 0.0,
        region_end_sec: float = -1.0,
    ) -> Tuple[Dict[str, Any]]:
        src = _validate_audio_latent(source_audio_latent, "source_audio_latent")
        src_duration = float(src.get("duration", 60.0))

        if operation == "lego":
            track_upper = primary_track.strip().upper()
            instruction = f"Generate the {track_upper} track based on the audio context:"
            selected_tracks = [primary_track]
        else:
            raw_list = [primary_track] + [
                t.strip().lower() for t in (additional_tracks or "").split(",") if t.strip()
            ]
            valid_set = set(TRACK_NAMES)
            selected_tracks: List[str] = []
            for t in raw_list:
                if t in valid_set and t not in selected_tracks:
                    selected_tracks.append(t)
            if not selected_tracks:
                selected_tracks = ["drums", "bass", "guitar"]
            classes_str = " | ".join(t.upper() for t in selected_tracks)
            instruction = f"Complete the input track with {classes_str}:"

        config: Dict[str, Any] = {
            "task_type": operation,
            "instruction": instruction,
            "track_name": primary_track,
            "track_classes": selected_tracks,
            "requires_base_model": True,
            "requires_flow_edit_support": False,
            "source_audio_latent": src,
            "repainting_start": float(region_start_sec),
            "repainting_end": None if region_end_sec < 0 else min(src_duration, float(region_end_sec)),
            "duration": src_duration,
        }
        return (config,)


class AceStepVocal2BGMTask:
    """Generate full instrumental BGM accompaniment around an isolated vocal track.

    Uses ACE-Step 1.5's ``complete`` task over a vocal source stem (requires a
    ``base`` DiT variant: ``acestep-v15-base`` or ``acestep-v15-xl-base``).
    """

    CATEGORY = "ACE-Step 1.5/Tasks"
    RETURN_TYPES = ("ACESTEP_TASK_CONFIG",)
    RETURN_NAMES = ("task_config",)
    FUNCTION = "build_vocal2bgm_task"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocal_audio_latent": (
                    "ACESTEP_AUDIO_LATENT",
                    {"tooltip": "Pre-encoded vocal track latent from ACE-Step 1.5 VAE Encode."},
                ),
                "add_drums": ("BOOLEAN", {"default": True, "tooltip": "Include drums in the generated BGM."}),
                "add_bass": ("BOOLEAN", {"default": True, "tooltip": "Include bass in the generated BGM."}),
                "add_guitar": ("BOOLEAN", {"default": True, "tooltip": "Include guitar in the generated BGM."}),
                "add_keyboard": ("BOOLEAN", {"default": True, "tooltip": "Include keyboard/piano in the generated BGM."}),
                "add_strings": ("BOOLEAN", {"default": False, "tooltip": "Include orchestral strings in the BGM."}),
                "add_synth": ("BOOLEAN", {"default": False, "tooltip": "Include synthesizers in the BGM."}),
                "add_percussion": ("BOOLEAN", {"default": False, "tooltip": "Include auxiliary percussion in the BGM."}),
                "add_brass": ("BOOLEAN", {"default": False, "tooltip": "Include brass section in the BGM."}),
                "add_woodwinds": ("BOOLEAN", {"default": False, "tooltip": "Include woodwinds in the BGM."}),
                "add_backing_vocals": ("BOOLEAN", {"default": False, "tooltip": "Include harmony/backing vocals."}),
                "add_fx": ("BOOLEAN", {"default": False, "tooltip": "Include atmospheric FX/textures."}),
            }
        }

    def build_vocal2bgm_task(
        self,
        vocal_audio_latent: Dict[str, Any],
        add_drums: bool,
        add_bass: bool,
        add_guitar: bool,
        add_keyboard: bool,
        add_strings: bool,
        add_synth: bool,
        add_percussion: bool,
        add_brass: bool,
        add_woodwinds: bool,
        add_backing_vocals: bool,
        add_fx: bool,
    ) -> Tuple[Dict[str, Any]]:
        src = _validate_audio_latent(vocal_audio_latent, "vocal_audio_latent")
        src_duration = float(src.get("duration", 60.0))

        flags = [
            ("drums", add_drums),
            ("bass", add_bass),
            ("guitar", add_guitar),
            ("keyboard", add_keyboard),
            ("strings", add_strings),
            ("synth", add_synth),
            ("percussion", add_percussion),
            ("brass", add_brass),
            ("woodwinds", add_woodwinds),
            ("backing_vocals", add_backing_vocals),
            ("fx", add_fx),
        ]
        selected = [name for name, enabled in flags if enabled]
        if not selected:
            selected = ["drums", "bass", "keyboard"]

        classes_str = " | ".join(t.upper() for t in selected)
        instruction = f"Complete the input track with {classes_str}:"

        config: Dict[str, Any] = {
            "task_type": "complete",
            "subtask": "vocal2bgm",
            "instruction": instruction,
            "track_classes": selected,
            "requires_base_model": True,
            "requires_flow_edit_support": False,
            "source_audio_latent": src,
            "duration": src_duration,
        }
        return (config,)


class AceStepMetadataControl:
    """Granular musical metadata and structured prompt composer (``ACESTEP_METADATA``)."""

    CATEGORY = "ACE-Step 1.5/LM"
    RETURN_TYPES = ("ACESTEP_METADATA", "STRING", "STRING")
    RETURN_NAMES = ("metadata", "composed_caption", "metadata_json")
    FUNCTION = "compose_metadata"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genres": (
                    "STRING",
                    {
                        "default": "indie rock, dream pop",
                        "multiline": False,
                        "tooltip": "Comma-separated musical genres and subgenres.",
                    },
                ),
                "mood": (
                    "STRING",
                    {
                        "default": "nostalgic, uplifting, cinematic",
                        "multiline": False,
                        "tooltip": "Emotional mood and atmosphere descriptors.",
                    },
                ),
                "instruments": (
                    "STRING",
                    {
                        "default": "shimmering electric guitar, warm analog bass, crisp live drums, lush synth pad",
                        "multiline": True,
                        "tooltip": "Detailed instrumentation and arrangement tags.",
                    },
                ),
                "vocal_style": (
                    "STRING",
                    {
                        "default": "expressive airy female vocals, light plate reverb",
                        "multiline": False,
                        "tooltip": "Vocal timbre and delivery style (ignored when instrumental=True).",
                    },
                ),
                "production_notes": (
                    "STRING",
                    {
                        "default": "wide stereo mix, clean mastering, dynamic chorus lift",
                        "multiline": False,
                        "tooltip": "Mixing, era, and production texture tags.",
                    },
                ),
                "bpm": (
                    "INT",
                    {
                        "default": 118,
                        "min": 0,
                        "max": 300,
                        "step": 1,
                        "tooltip": "Tempo in BPM (30-300; 0 = auto).",
                    },
                ),
                "key_note": (
                    KEY_NOTES,
                    {
                        "default": "A",
                        "tooltip": "Tonic note (or 'auto' to leave key unspecified).",
                    },
                ),
                "key_mode": (
                    KEY_MODES,
                    {
                        "default": "minor",
                        "tooltip": "Scale mode ('major' or 'minor'; ignored when key_note='auto').",
                    },
                ),
                "timesignature": (
                    TIMESIGNATURE_OPTIONS,
                    {
                        "default": "4",
                        "tooltip": "Time signature numerator (2=2/4, 3=3/4, 4=4/4, 6=6/8) or 'auto'.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": -1.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "Target duration in seconds (10-600 s; -1 = auto).",
                    },
                ),
                "vocal_language": (
                    VALID_LANGUAGES,
                    {
                        "default": "en",
                        "tooltip": "Vocal language code (50+ supported languages).",
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force purely instrumental output.",
                    },
                ),
            },
            "optional": {
                "lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional structured lyrics ([Verse], [Chorus], [Bridge], [Outro]).",
                    },
                ),
            },
        }

    def compose_metadata(
        self,
        genres: str,
        mood: str,
        instruments: str,
        vocal_style: str,
        production_notes: str,
        bpm: int,
        key_note: str,
        key_mode: str,
        timesignature: str,
        duration: float,
        vocal_language: str,
        instrumental: bool,
        lyrics: str = "",
    ) -> Tuple[Dict[str, Any], str, str]:
        segments = [
            s.strip()
            for s in (
                genres,
                mood,
                instruments,
                "" if instrumental else vocal_style,
                production_notes,
            )
            if s and s.strip()
        ]
        if instrumental:
            segments.append("instrumental, no vocals")
        composed_caption = ", ".join(segments)

        resolved_keyscale = "" if key_note == "auto" else f"{key_note} {key_mode}"
        resolved_bpm = _coerce_optional_int(bpm, 30, 300) if bpm > 0 else None
        resolved_timesig = _normalize_timesig(timesignature)
        resolved_duration = _coerce_optional_duration(duration)
        resolved_lang = "unknown" if instrumental else vocal_language
        resolved_lyrics = "[Instrumental]" if instrumental else (lyrics or "").strip()

        payload: Dict[str, Any] = {
            "caption": composed_caption,
            "lyrics": resolved_lyrics,
            "bpm": resolved_bpm,
            "keyscale": resolved_keyscale,
            "timesignature": resolved_timesig,
            "duration": resolved_duration,
            "vocal_language": resolved_lang,
            "instrumental": bool(instrumental),
            "tags": {
                "genres": genres.strip(),
                "mood": mood.strip(),
                "instruments": instruments.strip(),
                "vocal_style": "" if instrumental else vocal_style.strip(),
                "production_notes": production_notes.strip(),
            },
        }
        return (payload, composed_caption, json.dumps(payload, indent=2, ensure_ascii=False))


class AceStepQueryRewriter:
    """Advanced Query Rewriting and prompt/metadata expansion powered by ACE-Step's 5Hz LM."""

    CATEGORY = "ACE-Step 1.5/LM"
    RETURN_TYPES = ("STRING", "STRING", "ACESTEP_METADATA", "STRING")
    RETURN_NAMES = ("rewritten_caption", "structured_lyrics", "metadata", "metadata_json")
    FUNCTION = "rewrite_query"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lm_model": (
                    "ACESTEP_LM",
                    {"tooltip": "5Hz LM handle from ACE-Step 1.5 LM Loader."},
                ),
                "rewrite_mode": (
                    ["inspiration_simple", "format_enhance", "metadata_guided_rewrite"],
                    {
                        "default": "inspiration_simple",
                        "tooltip": (
                            "'inspiration_simple': expand a brief idea into full caption, lyrics, and metadata; "
                            "'format_enhance': rewrite and enrich an existing caption/lyrics pair into ACE-Step style; "
                            "'metadata_guided_rewrite': rewrite caption while strictly locking supplied metadata fields."
                        ),
                    },
                ),
                "query_or_caption": (
                    "STRING",
                    {
                        "default": "a bittersweet midnight synthwave drive through rain-soaked neon streets",
                        "multiline": True,
                        "tooltip": "Natural-language song idea or rough style caption to rewrite.",
                    },
                ),
                "lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional draft lyrics to format or preserve.",
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Rewrite for an instrumental-only composition.",
                    },
                ),
                "vocal_language": (
                    VALID_LANGUAGES,
                    {
                        "default": "en",
                        "tooltip": "Preferred vocal language ('unknown' lets the 5Hz LM choose).",
                    },
                ),
                "preserve_user_lyrics": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When draft lyrics are supplied, keep them instead of overwriting with generated lyrics.",
                    },
                ),
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature for query rewriting.",
                    },
                ),
                "lm_top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Top-K cutoff (0 disables Top-K).",
                    },
                ),
                "lm_top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Top-P nucleus sampling threshold.",
                    },
                ),
                "lm_repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Repetition penalty (1.0 = disabled).",
                    },
                ),
                "use_constrained_decoding": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enforce valid musical metadata fields via FSM constrained decoding.",
                    },
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload 5Hz LM and purge CUDA/ComfyUI caches immediately after rewriting.",
                    },
                ),
            },
            "optional": {
                "metadata": (
                    "ACESTEP_METADATA",
                    {"tooltip": "Optional metadata constraints from ACE-Step 1.5 Metadata Control."},
                ),
            },
        }

    def rewrite_query(
        self,
        lm_model: AceStepLMHandle,
        rewrite_mode: str,
        query_or_caption: str,
        lyrics: str,
        instrumental: bool,
        vocal_language: str,
        preserve_user_lyrics: bool,
        lm_temperature: float,
        lm_top_k: int,
        lm_top_p: float,
        lm_repetition_penalty: float,
        use_constrained_decoding: bool,
        unload_after_run: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, Dict[str, Any], str]:
        meta_in = metadata if isinstance(metadata, dict) else {}
        base_caption = (query_or_caption or "").strip() or str(meta_in.get("caption", "")).strip()
        is_inst = instrumental or bool(meta_in.get("instrumental", False))
        base_lyrics = "[Instrumental]" if is_inst else ((lyrics or "").strip() or str(meta_in.get("lyrics", "")).strip())
        lang = "unknown" if is_inst else (
            vocal_language if vocal_language != "unknown" else str(meta_in.get("vocal_language", "unknown"))
        )

        user_meta: Dict[str, Any] = {}
        if meta_in.get("bpm") is not None:
            user_meta["bpm"] = meta_in["bpm"]
        if meta_in.get("keyscale"):
            user_meta["keyscale"] = meta_in["keyscale"]
        if meta_in.get("timesignature"):
            user_meta["timesignature"] = meta_in["timesignature"]
        if meta_in.get("duration") is not None:
            user_meta["duration"] = int(round(float(meta_in["duration"])))
        if lang and lang != "unknown":
            user_meta["language"] = lang

        top_k_arg = lm_top_k if lm_top_k > 0 else None
        top_p_arg = lm_top_p if 0.0 < lm_top_p < 1.0 else None

        out_caption = base_caption
        out_lyrics = base_lyrics
        out_bpm = _coerce_optional_int(meta_in.get("bpm"), 30, 300)
        out_keyscale = str(meta_in.get("keyscale", ""))
        out_timesig = _normalize_timesig(meta_in.get("timesignature"))
        out_duration = _coerce_optional_duration(meta_in.get("duration"))
        out_lang = lang

        try:
            llm_handler = lm_model.ensure_loaded()
            if rewrite_mode == "inspiration_simple":
                result_meta, status = llm_handler.create_sample_from_query(
                    query=base_caption,
                    instrumental=is_inst,
                    vocal_language=None if lang == "unknown" else lang,
                    temperature=lm_temperature,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    repetition_penalty=lm_repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                )
                logger.info("[ACE-Step 1.5] QueryRewriter (inspiration_simple): %s", status)
            else:
                guided_meta = user_meta if (rewrite_mode == "metadata_guided_rewrite" and user_meta) else (user_meta or None)
                result_meta, status = llm_handler.format_sample_from_input(
                    caption=base_caption,
                    lyrics=base_lyrics,
                    user_metadata=guided_meta,
                    temperature=lm_temperature,
                    top_k=top_k_arg,
                    top_p=top_p_arg,
                    repetition_penalty=lm_repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                )
                logger.info("[ACE-Step 1.5] QueryRewriter (%s): %s", rewrite_mode, status)

            if isinstance(result_meta, dict) and result_meta:
                if result_meta.get("caption"):
                    out_caption = str(result_meta["caption"]).strip()
                if not is_inst:
                    if not (preserve_user_lyrics and base_lyrics):
                        if result_meta.get("lyrics"):
                            out_lyrics = str(result_meta["lyrics"]).strip()
                if out_bpm is None or rewrite_mode != "metadata_guided_rewrite":
                    parsed_bpm = _coerce_optional_int(result_meta.get("bpm"), 30, 300)
                    if parsed_bpm is not None:
                        out_bpm = parsed_bpm
                if (not out_keyscale or rewrite_mode != "metadata_guided_rewrite") and result_meta.get("keyscale"):
                    out_keyscale = str(result_meta["keyscale"]).strip()
                if (not out_timesig or rewrite_mode != "metadata_guided_rewrite") and result_meta.get("timesignature"):
                    parsed_ts = _normalize_timesig(result_meta.get("timesignature"))
                    if parsed_ts:
                        out_timesig = parsed_ts
                if (out_duration is None or rewrite_mode != "metadata_guided_rewrite") and result_meta.get("duration"):
                    parsed_dur = _coerce_optional_duration(result_meta.get("duration"))
                    if parsed_dur is not None:
                        out_duration = parsed_dur
                if not is_inst and result_meta.get("language"):
                    out_lang = str(result_meta["language"]).strip()
        finally:
            if unload_after_run:
                logger.info("[ACE-Step 1.5] Unloading 5Hz LM after query rewriting (unload_after_run=True).")
                lm_model.unload(release_reference=True)
            else:
                purge_vram(aggressive=False)

        out_payload: Dict[str, Any] = {
            "caption": out_caption,
            "lyrics": out_lyrics,
            "bpm": out_bpm,
            "keyscale": out_keyscale,
            "timesignature": out_timesig,
            "duration": out_duration,
            "vocal_language": out_lang,
            "instrumental": bool(is_inst),
        }
        return (out_caption, out_lyrics, out_payload, json.dumps(out_payload, indent=2, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "AceStepReferenceAudio": AceStepReferenceAudio,
    "AceStepCoverTask": AceStepCoverTask,
    "AceStepRepaintTask": AceStepRepaintTask,
    "AceStepTrackExtractTask": AceStepTrackExtractTask,
    "AceStepMultiTrackTask": AceStepMultiTrackTask,
    "AceStepVocal2BGMTask": AceStepVocal2BGMTask,
    "AceStepMetadataControl": AceStepMetadataControl,
    "AceStepQueryRewriter": AceStepQueryRewriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepReferenceAudio": "ACE-Step 1.5 Reference Audio",
    "AceStepCoverTask": "ACE-Step 1.5 Cover & Flow-Edit Task",
    "AceStepRepaintTask": "ACE-Step 1.5 Repaint & Edit Task",
    "AceStepTrackExtractTask": "ACE-Step 1.5 Track Separation (Extract)",
    "AceStepMultiTrackTask": "ACE-Step 1.5 Multi-Track Task (Lego / Complete)",
    "AceStepVocal2BGMTask": "ACE-Step 1.5 Vocal2BGM Task",
    "AceStepMetadataControl": "ACE-Step 1.5 Metadata Control",
    "AceStepQueryRewriter": "ACE-Step 1.5 Query Rewriter (5Hz LM)",
}
