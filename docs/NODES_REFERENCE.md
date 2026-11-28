# Node Reference — ComfyUI-Bao-Ace_Step_1_5

Complete reference for all 19 custom nodes in **ComfyUI-Bao-Ace_Step_1_5**, organized by ComfyUI menu category. Every node enforces strict 12 GB VRAM isolation with `unload_after_run=True` by default.

---

## Custom Port Types

| Type | Residence | Description |
| --- | --- | --- |
| `ACESTEP_LM` | Lazy Handle | Configured 5Hz Language Model handle (`0.6B`, `1.7B`, `4B`, PyTorch `pt` backend). |
| `ACESTEP_DIT` | Lazy Handle | Configured DiT handle (`2B` or `XL 4B`, `base`/`sft`/`turbo`) + `Qwen3-Embedding-0.6B`. |
| `ACESTEP_VAE` | Lazy Handle | Configured `AutoencoderOobleck` handle (`vae` or `scragvae`). |
| `ACESTEP_METADATA` | CPU Dict | Structured musical metadata (`caption`, `lyrics`, `bpm`, `keyscale`, `timesignature`, `duration`, `vocal_language`, `instrumental`). |
| `ACESTEP_PLAN` | CPU Dict | Full LM planning payload including 5Hz semantic `audio_codes` and resolved metadata. |
| `ACESTEP_AUDIO_LATENT` | CPU Dict | Pre-encoded 25Hz latents (`[1, T, 64]`) plus resampled 48 kHz stereo waveform reference. |
| `ACESTEP_REFERENCE_AUDIO` | CPU Dict | Cropped 25Hz reference excerpt for style/timbre conditioning. |
| `ACESTEP_TASK_CONFIG` | CPU Dict | Task-specific instruction and conditioning bundle (`cover`, `cover-nofsq`, `repaint`, `extract`, `lego`, `complete`). |
| `ACESTEP_LATENT` | CPU Dict | Synthesized 25Hz latents (`[B, T, 64]`), optional `alignment_data` (CPU), and optional `repaint_splice` data. |
| `AUDIO` | CPU Dict | Native ComfyUI audio format: `{"waveform": Tensor[B, 2, T] (float32, CPU), "sample_rate": 48000}`. |

---

## 1. Loaders (`ACE-Step 1.5/Loaders`)

> **Model Directory Resolution (`extra_model_paths.yaml`):** If `ComfyUI/extra_model_paths.yaml` defines an `acestep:` folder, all loaders automatically prioritize that path for scanning, downloading (DiT, 5Hz LM, VAE, `Qwen3-Embedding-0.6B`), and saving quantized `*_int8_convrot` checkpoints—without creating an empty `ComfyUI/models/acestep/` folder.

### `AceStepLMLoader` — ACE-Step 1.5 LM Loader (5Hz)
Prepares a lazy 5Hz Language Model handle using the native PyTorch (`pt`) backend for full Python 3.12/3.13 compatibility.
- **Inputs (Required):**
  - `model_path`: Checkpoint in your configured `acestep` models folder (`extra_model_paths.yaml` or `ComfyUI/models/acestep/`; `acestep-5Hz-lm-0.6B`, `1.7B`, `4B`, or local quantized `*_int8_convrot` folder). Selecting a `*_int8_convrot` folder directly auto-detects INT8 even if `precision` is set to `bf16`/`fp16`.
  - `device`: `auto_offload` *(12 GB recommended)*, `cuda`, or `cpu`.
  - `precision`: `int8_dynamic` *(12 GB recommended for 4B)*, `bf16`, or `fp16`.
  - `save_quantized`: Export INT8 weights + scales to `<model_path>_int8_convrot/model_int8.safetensors` on first run (default `False`). Existing `*_int8_convrot` folders are checked **before** any download, so the original folder is not required and will not be re-downloaded.
  - `auto_download`: Fetch missing official Hugging Face weights automatically into the configured `acestep` directory (default `True`).
- **Inputs (Optional):**
  - `overwrite_quantized`: Re-quantize and overwrite an existing `*_int8_convrot` folder (default `False`).
  - `delete_original_after_quantize`: Automatically remove the original full-precision folder inside your configured `acestep` directory after `save_quantized=True` writes and verifies the INT8 export (default `False`).
- **Outputs:** `lm_model` (`ACESTEP_LM`).

### `AceStepDiTLoader` — ACE-Step 1.5 DiT Loader (2B / XL 4B)
Prepares a lazy DiT + `Qwen3-Embedding-0.6B` handle without loading the VAE.
- **Inputs (Required):**
  - `model_path`: `acestep-v15-xl-base`, `acestep-v15-xl-sft`, `acestep-v15-xl-turbo`, `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo`, or local quantized `*_int8_convrot` folder (auto-detected as INT8 regardless of `precision`).
  - `precision`: `int8_dynamic` *(12 GB recommended for XL 4B)*, `bf16`, or `fp16`.
  - `save_quantized`: Save quantized decoder layers to `<model>_int8_convrot/model_int8.safetensors` inside the configured `acestep` directory (default `False`). Existing `*_int8_convrot` folders are checked **before** any download so you can delete the original full-precision folder without triggering re-downloads.
  - `offload_dit_to_cpu`: Enable block-by-block CPU offloading when free VRAM drops below `vram_offload_threshold_gb` (default `True`).
  - `vram_offload_threshold_gb`: Free VRAM safety floor in GB (default `1.5`).
  - `use_flash_attention`: Prefer Flash Attention 2 when installed, falling back to SDPA/eager (default `True`).
  - `auto_download`: Auto-fetch missing DiT weights and isolated `Qwen3-Embedding-0.6B/*` files into the configured `acestep` directory without pulling unrelated base models (default `True`).
- **Inputs (Optional):**
  - `overwrite_quantized`: Re-quantize and overwrite an existing `*_int8_convrot` folder (default `False`).
  - `delete_original_after_quantize`: Automatically remove the original full-precision folder inside your configured `acestep` directory after `save_quantized=True` writes and verifies the INT8 export (default `False`).
- **Outputs:** `dit_model` (`ACESTEP_DIT`).

### `AceStepVAELoader` — ACE-Step 1.5 VAE Loader
Prepares a lazy `AutoencoderOobleck` handle for 48 kHz stereo encoding and decoding.
- **Inputs (Required):** `model_path` (`vae`, `scragvae`, or local folder), `device` (`cuda`, `cpu`), `precision` (`fp16`, `bf16`, `fp32`), `auto_download` (`True`, downloads into the configured `acestep` directory).
- **Outputs:** `vae_model` (`ACESTEP_VAE`).

---

## 2. VAE Nodes (`ACE-Step 1.5/VAE`)

### `AceStepVAEEncode` — ACE-Step 1.5 VAE Encode
Resamples ComfyUI `AUDIO` to 48 kHz stereo and encodes it into 25Hz latents using overlap-discard tiling, then purges the VAE from VRAM before any DiT node runs.
- **Inputs (Required):** `vae_model`, `audio`, `use_tiled_encode` (`True`), `chunk_seconds` (`15.0` for 12 GB VRAM), `overlap_seconds` (`2.0`), `normalize_peak` (`True`), `fallback_to_cpu_on_oom` (`True`), `unload_after_run` (`True`).
- **Inputs (Optional):** `trim_start_sec` (`0.0`), `trim_end_sec` (`-1.0`).
- **Outputs:** `audio_latent` (`ACESTEP_AUDIO_LATENT`), `duration_sec` (`FLOAT`).

### `AceStepVAEDecode` — ACE-Step 1.5 VAE Decode
Decodes 25Hz `ACESTEP_LATENT` into native ComfyUI 48 kHz stereo `AUDIO` using overlap-discard tiling.
- **Inputs (Required):** `vae_model`, `latent`, `use_tiled_decode` (`True`), `chunk_size` (`256` frames ~= 10.2 s), `overlap` (`64`), `offload_wav_to_cpu` (`True`), `fallback_to_cpu_on_oom` (`True`), `unload_after_run` (`True`).
- **Inputs (Optional):** `apply_repaint_splice` (`True`, restores original pre-VAE audio outside the repainted region in CPU), `repaint_crossfade_sec` (`-1.0` = automatic from `AceStepRepaintTask`).
- **Outputs:** `audio` (`AUDIO`).

---

## 3. Language Model, Metadata & Query Rewriting (`ACE-Step 1.5/LM`)

### `AceStepMetadataControl` — ACE-Step 1.5 Metadata Control
Composes validated musical captions and structured metadata (`ACESTEP_METADATA`) with separate controls for `genres`, `mood`, `instruments`, `vocal_style`, `production_notes`, `bpm`, `key_note` + `key_mode`, `timesignature`, `duration`, `vocal_language`, `instrumental`, and optional `lyrics`.
- **Outputs:** `metadata` (`ACESTEP_METADATA`), `composed_caption` (`STRING`), `metadata_json` (`STRING`).

### `AceStepQueryRewriter` — ACE-Step 1.5 Query Rewriter (5Hz LM)
Expands or formats user queries via the 5Hz LM (`inspiration_simple`, `format_enhance`, `metadata_guided_rewrite`) while optionally locking user-supplied metadata or preserving draft lyrics.
- **Key Parameters:** `lm_model`, `rewrite_mode`, `query_or_caption`, `lyrics`, `instrumental`, `vocal_language`, `preserve_user_lyrics` (`True`), `lm_temperature` (`0.85`), `lm_top_k` (`0`), `lm_top_p` (`0.9`), `lm_repetition_penalty` (`1.0`), `use_constrained_decoding` (`True`), `unload_after_run` (`True`), optional `metadata`.
- **Outputs:** `rewritten_caption` (`STRING`), `structured_lyrics` (`STRING`), `metadata` (`ACESTEP_METADATA`), `metadata_json` (`STRING`).

### `AceStepLMPlanner` — ACE-Step 1.5 LM Planner (5Hz)
Runs Simple Mode query expansion, Custom CoT metadata/caption refinement, and 5Hz semantic audio code generation (`generate_with_stop_condition`).
- **Modes:** `simple_query_rewrite`, `custom_cot_and_codes`, `custom_format_and_codes`, `bypass`.
- **12 GB Defaults:** `batch_size=1`, `generate_audio_codes=True`, `use_constrained_decoding=True`, `unload_after_run=True`.
- **Optional Inputs:** `metadata` (`ACESTEP_METADATA`), CoT flags, and LM sampling hyperparameters (`lm_cfg_scale=2.0`, `lm_temperature=0.85`, `lm_negative_prompt="NO USER INPUT"`).
- **Outputs:** `lm_plan` (`ACESTEP_PLAN`), `caption`, `lyrics`, `metadata_json`, `audio_codes`.

---

## 4. Audio Tasks & Conditioning (`ACE-Step 1.5/Tasks`)

### `AceStepReferenceAudio` — ACE-Step 1.5 Reference Audio
Extracts a style/timbre reference window (`start_offset_sec`, `max_duration_sec`, default `30.0` s) from `ACESTEP_AUDIO_LATENT` for zero-VAE injection into `AceStepSampler`.
- **Outputs:** `reference_audio` (`ACESTEP_REFERENCE_AUDIO`).

### `AceStepCoverTask` — ACE-Step 1.5 Cover & Flow-Edit Task
Configures `cover`, `cover-nofsq`, and optional **Flow-Edit morph overlay** (`enable_flow_edit_morph`, `flow_edit_source_caption`, `flow_edit_source_lyrics`, `flow_edit_n_min`, `flow_edit_n_max`, `flow_edit_n_avg`).
- **Key Parameters:** `source_audio_latent`, `cover_mode`, `audio_cover_strength` (`0.8`), `cover_noise_strength` (`0.0`), optional `source_audio_codes`.
- **Outputs:** `task_config` (`ACESTEP_TASK_CONFIG`).

### `AceStepRepaintTask` — ACE-Step 1.5 Repaint & Edit Task
Configures temporal audio inpainting over `[repainting_start, repainting_end]` with `repaint_mode` (`balanced`, `conservative`, `aggressive`), `repaint_strength` (`0.5`), `chunk_mask_mode` (`explicit`, `auto`), and `enable_waveform_splice` (`True`).
- **Outputs:** `task_config` (`ACESTEP_TASK_CONFIG`).

### `AceStepTrackExtractTask` — ACE-Step 1.5 Track Separation (Extract)
Configures single-stem extraction (`vocals`, `backing_vocals`, `drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`). **Requires a `base` DiT model** (`acestep-v15-base` or `acestep-v15-xl-base`).
- **Outputs:** `task_config` (`ACESTEP_TASK_CONFIG`).

### `AceStepMultiTrackTask` — ACE-Step 1.5 Multi-Track Task (Lego / Complete)
Adds a single instrument stem (`lego`) or completes a multi-instrument arrangement (`complete` with `primary_track` + comma-separated `additional_tracks`). **Requires a `base` DiT model**.
- **Outputs:** `task_config` (`ACESTEP_TASK_CONFIG`).

### `AceStepVocal2BGMTask` — ACE-Step 1.5 Vocal2BGM Task
Builds full instrumental backing (`drums`, `bass`, `guitar`, `keyboard`, `strings`, `synth`, `percussion`, `brass`, `woodwinds`, `backing_vocals`, `fx`) around an encoded vocal stem via `complete`. **Requires a `base` DiT model**.
- **Outputs:** `task_config` (`ACESTEP_TASK_CONFIG`).

---

## 5. Sampling (`ACE-Step 1.5/Sampling`)

### `AceStepSampler` — ACE-Step 1.5 DiT Sampler (25Hz Latents)
Executes DiT flow-matching diffusion across all Phase 1 and Phase 2 tasks while isolating `Qwen3-Embedding-0.6B`, never loading the VAE, and applying block-level CPU offloading when needed.
- **Required Parameters:** `dit_model`, `inference_steps` (`8` for turbo, `32–60` for base/sft), `guidance_scale` (`1.0` for turbo, `7.0` for base/sft), `shift` (`3.0`), `infer_method` (`ode`, `sde`), `sampler_mode` (`euler`, `heun`), `use_adg`, `cfg_interval_start`, `cfg_interval_end`, `audio_cover_strength`, `seed`, `batch_size` (`1` for 12 GB), `unload_after_run` (`True`).
- **Optional Inputs:** `lm_plan`, `task_config`, `reference_audio`, `metadata`, `caption_override`, `lyrics_override`, `instrumental`, `vocal_language`, `bpm_override`, `keyscale_override`, `timesignature_override`, `duration_override`, `audio_codes_override`, `capture_alignment_data` (`True` when feeding LRC or Quality Scoring), `latent_shift`, `latent_rescale`.
- **Outputs:** `latent` (`ACESTEP_LATENT`), `generation_info` (`STRING`).

---

## 6. Analysis, LRC & Quality Scoring (`ACE-Step 1.5/Analysis`)

### `AceStepAudioTokenizer` — ACE-Step 1.5 Audio Tokenizer (5Hz Codes)
Converts `ACESTEP_AUDIO_LATENT` into serialized 5Hz semantic tokens (`<|audio_code_N|>...`) via the DiT FSQ tokenizer, then unloads the DiT cleanly.
- **Outputs:** `audio_codes` (`STRING`), `token_count` (`INT`).

### `AceStepAudioUnderstanding` — ACE-Step 1.5 Audio Understanding (5Hz LM)
Interprets 5Hz `audio_codes` using the 5Hz LM to reverse-engineer caption, lyrics, BPM, keyscale, time signature, duration, and vocal language.
- **Outputs:** `caption`, `lyrics`, `metadata` (`ACESTEP_METADATA`), `understanding_json`.

### `AceStepLRCGenerator` — ACE-Step 1.5 LRC Generator
Extracts DiT cross-attention matrices and aligns lyrics via bidirectional consensus denoising + DTW (`MusicStampsAligner`). Optionally writes sequential `.lrc` files into `ComfyUI/output/`.
- **Prerequisite:** Enable `capture_alignment_data=True` on `AceStepSampler`.
- **Outputs:** `lrc_text` (`STRING`), `timestamps_json` (`STRING`), `saved_lrc_path` (`STRING`).

### `AceStepQualityScorer` — ACE-Step 1.5 Quality Scorer (DiT + LM PMI)
Computes DiT lyric alignment (`get_lyric_score`) and 5Hz LM Pointwise Mutual Information + reward scoring (`calculate_pmi_score_per_condition`, `calculate_reward_score`) strictly sequentially: DiT runs first and is completely unloaded before LM loads.
- **Outputs:** `overall_score` (`FLOAT`), `dit_lyric_score` (`FLOAT`), `lm_alignment_score` (`FLOAT`), `lm_pmi_reward` (`FLOAT`), `report_json` (`STRING`).
