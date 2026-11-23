# Audio Tasks, Editing & Metadata Control Guide

This guide explains how to use the Phase 2 task nodes in **ComfyUI-Bao-Ace_Step_1_5** and which DiT model checkpoints support each task.

---

## Checkpoint Compatibility Matrix

| Task / Capability | `turbo` / `xl-turbo` | `sft` / `xl-sft` | `base` / `xl-base` | Recommended Steps / CFG |
| --- | :---: | :---: | :---: | --- |
| Text-to-Music & Simple Mode | Yes | Yes | Yes | Turbo: `8 / 1.0` · SFT/Base: `32–50 / 7.0` |
| Reference Audio Input | Yes | Yes | Yes | Turbo: `8 / 1.0` · SFT/Base: `32–50 / 7.0` |
| Cover (`cover` / `cover-nofsq`) | Yes | Yes | Yes | Turbo: `8 / 1.0` · SFT/Base: `32–50 / 7.0` |
| Flow-Edit Smooth Morph (`flow_edit_morph`) | No | Yes | Yes | `32–50 / 7.0` |
| Repaint & Edit (`repaint`) | Yes | Yes | Yes | Turbo: `8 / 1.0` · SFT/Base: `32–50 / 7.0` |
| Track Separation (`extract`) | No | No | **Yes** | `40–60 / 7.0` |
| Multi-Track (`lego` / `complete`) | No | No | **Yes** | `40–60 / 7.0` |
| Vocal2BGM (`complete`) | No | No | **Yes** | `40–60 / 7.0` |

> `AceStepSampler` validates task compatibility before running diffusion and raises a clear message if a `turbo` checkpoint is wired into a `base`-only task or Flow-Edit.

---

## 1. Reference Audio Input (`workflows/01_reference_audio_input.json`)

**Goal:** Transfer timbre, vocal color, mixing sheen, or acoustic texture from an existing audio clip into a brand-new composition.

1. Connect `LoadAudio` → `AceStepVAEEncode` (`unload_after_run=True`).
2. Connect `audio_latent` → `AceStepReferenceAudio` (`max_duration_sec=30.0`, `start_offset_sec=0.0`).
3. Wire `reference_audio` into `AceStepSampler`'s `reference_audio` input alongside your caption/lyrics or `lm_plan`.
4. **Why it fits in 12 GB VRAM:** `AceStepVAEEncode` finishes and unloads the VAE before `AceStepSampler` loads DiT XL; the pre-encoded latent is injected directly into `infer_refer_latent` without reloading the VAE.

---

## 2. Cover Generation & Flow-Edit (`workflows/02_cover_generation.json`)

**Goal:** Re-arrange an existing song in a new genre, instrumentation, or vocal style while preserving melody and phrasing.

- **`cover` mode:** Uses ACE-Step's FSQ semantic structure tokens. Adjust `audio_cover_strength` (`0.7–1.0` for faithful covers; `0.3–0.6` for creative remixes).
- **`cover-nofsq` mode:** Uses continuous source latents for smoother timbral continuity.
- **Flow-Edit Morph (`enable_flow_edit_morph=True`):** Uses paired source/target velocity integration ($V_{\text{tar}} - V_{\text{src}}$). Provide `flow_edit_source_caption` (and optional `flow_edit_source_lyrics`) describing the *original* track, and put the *target* style in `caption_override` / `metadata`. Requires `base`, `sft`, `xl-base`, or `xl-sft`.

---

## 3. Repaint & Edit (`workflows/03_repaint_and_edit.json`)

**Goal:** Regenerate a specific temporal window (`[repainting_start, repainting_end]`, e.g., fix a vocal line or replace a 15-second chorus) while keeping the rest of the track untouched.

- **Modes:**
  - `conservative`: Maximum preservation of surrounding audio (`injection_ratio=1.0`, 25-frame latent crossfade, `50 ms` waveform splice crossfade).
  - `balanced`: Controlled by `repaint_strength` (`0.0` = conservative, `1.0` = aggressive). Default `0.5` works well for lyric edits and instrumentation fixes.
  - `aggressive`: Pure diffusion inside the masked region without step injection.
- **Waveform Splice (`enable_waveform_splice=True`):** Preserves the exact original user waveform outside `[repainting_start, repainting_end]` during `AceStepVAEDecode` on CPU with smooth linear edge ramps—eliminating VAE round-trip degradation in unedited segments.

---

## 4. Track Separation — Extract (`workflows/04_track_separation.json`)

**Goal:** Isolate one of the 12 supported stems (`vocals`, `backing_vocals`, `drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`) from a mixed song.

1. Use `acestep-v15-base` or `acestep-v15-xl-base` in `AceStepDiTLoader`.
2. Encode the mix with `AceStepVAEEncode`.
3. Select the desired `track_name` in `AceStepTrackExtractTask` and connect `task_config` → `AceStepSampler` (`inference_steps=40–60`, `guidance_scale=7.0`).

---

## 5. Multi-Track Generation — Lego & Complete (`workflows/05_multi_track_generation.json`)

**Goal:** Layer new instruments onto an existing stem or expand a sketch into a full arrangement. Requires `acestep-v15-base` or `acestep-v15-xl-base`.

- **`lego` operation:** Synthesizes a single `primary_track` (e.g., `guitar` or `strings`) over the source audio context, optionally bounded to `[region_start_sec, region_end_sec]`.
- **`complete` operation:** Completes the input track using `primary_track` plus comma-separated `additional_tracks` (e.g., `drums, bass, keyboard`).

---

## 6. Vocal2BGM (`workflows/06_vocal2bgm.json`)

**Goal:** Turn an a-cappella or isolated vocal recording into a full studio arrangement automatically.

1. Use `acestep-v15-base` or `acestep-v15-xl-base`.
2. Encode the vocal stem via `AceStepVAEEncode`.
3. Toggle the desired backing instruments (`add_drums`, `add_bass`, `add_guitar`, `add_keyboard`, `add_strings`, `add_synth`, etc.) in `AceStepVocal2BGMTask`.
4. Describe the target genre/mood in `caption_override` or `AceStepMetadataControl` and keep the matching lyrics in `lyrics_override` for best rhythmic alignment.

---

## 7. Granular Metadata Control & Query Rewriting

- **`AceStepMetadataControl` (`workflows/07_metadata_control.json`):** Separates musical direction into `genres`, `mood`, `instruments`, `vocal_style`, `production_notes`, `bpm`, `key_note` + `key_mode`, `timesignature`, `duration`, `vocal_language`, and `instrumental`. Outputs `ACESTEP_METADATA`, which plugs into `AceStepLMPlanner`, `AceStepQueryRewriter`, or directly into `AceStepSampler`.
- **`AceStepQueryRewriter` (`workflows/09_query_rewriting.json`):** Uses the 5Hz LM in three dedicated modes:
  - `inspiration_simple`: Expands a brief concept into a full studio prompt, structured lyrics, and metadata.
  - `format_enhance`: Polishes rough tags/lyrics into ACE-Step's native conditioning format.
  - `metadata_guided_rewrite`: Expands the caption while strictly preserving user-locked BPM, key, time signature, and duration from `AceStepMetadataControl`.
