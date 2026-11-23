# Audio Understanding, LRC Generation & Quality Scoring Guide

This guide covers the analytical nodes under `ACE-Step 1.5/Analysis` and how they execute sequentially within a **12 GB VRAM** budget.

---

## 1. Audio Understanding (`workflows/10_audio_understanding.json`)

Reverse-engineers any audio file into structured musical metadata (caption, lyrics, BPM, keyscale, time signature, vocal language, and duration) that can be inspected or fed directly into Cover, Repaint, or Query Rewriting workflows.

### Strict 3-Stage VRAM Pipeline (<= 12 GB)
1. **Stage 1 (VAE only):** `LoadAudio` → `AceStepVAEEncode` (`unload_after_run=True`) converts audio to 25Hz latents and frees the VAE.
2. **Stage 2 (DiT FSQ Tokenizer only):** `AceStepAudioTokenizer` (`unload_after_run=True`) quantizes the 25Hz latents into 5Hz `<|audio_code_N|>` tokens and frees the DiT completely.
3. **Stage 3 (5Hz LM only):** `AceStepAudioUnderstanding` (`unload_after_run=True`, recommended `lm_temperature=0.3`) decodes the 5Hz codes into:
   - `caption` (`STRING`)
   - `lyrics` (`STRING`)
   - `metadata` (`ACESTEP_METADATA`, directly connectable to `AceStepSampler` or `AceStepLMPlanner`)
   - `understanding_json` (`STRING`)

---

## 2. LRC Generation (`workflows/11_lrc_generation.json`)

Produces synchronized `[mm:ss.xx]` `.lrc` karaoke/subtitle timestamps using DiT decoder cross-attention matrices, bidirectional consensus denoising, and Dynamic Time Warping (`MusicStampsAligner`).

### Setup & Best Practices
1. Enable **`capture_alignment_data=True`** on `AceStepSampler`. This captures `encoder_hidden_states`, `encoder_attention_mask`, `context_latents`, and `lyric_token_ids` and immediately offloads them to CPU RAM inside `ACESTEP_LATENT` (zero VRAM retained).
2. Connect `dit_model` and `latent` to `AceStepLRCGenerator`.
3. Match `inference_steps` and `seed` with the sampler for deterministic alignment (`inference_steps=8` for turbo, `32` for SFT/base).
4. Set `save_lrc_file=True` and `filename_prefix` to automatically export `<prefix>_0001.lrc` inside `ComfyUI/output/`, while also inspecting `lrc_text` and `timestamps_json` in the graph.

---

## 3. Quality Scoring (`workflows/12_quality_scoring.json`)

Evaluates generated music objectively using both ACE-Step 1.5 scoring engines:

1. **DiT Lyric Alignment Scoring (`dit_lyric_score`, `lm_alignment_score`):**
   - Uses `handler.get_lyric_score()` (`MusicLyricScorer`) over cross-attention matrices to measure how cleanly sung vocals align with the target lyrics.
   - Requires `dit_model` connected and `capture_alignment_data=True` on `AceStepSampler`.
2. **5Hz LM PMI & Reward Scoring (`lm_pmi_reward`):**
   - Computes Pointwise Mutual Information (PMI) between the generated 5Hz audio codes and each requested condition (caption, lyrics, BPM, keyscale, time signature, language) via `calculate_pmi_score_per_condition` and aggregates them with `calculate_reward_score`.
   - If `audio_codes` is not supplied manually, `AceStepQualityScorer` automatically tokenizes `latent["samples"]` during Stage 1 before unloading the DiT!

### Why Quality Scoring Fits in 12 GB VRAM
Even when both `dit_model` (4B XL) and `lm_model` (4B) are connected simultaneously to `AceStepQualityScorer`:
1. The node loads **only** the DiT, computes `get_lyric_score` + 5Hz tokenization, and **immediately unloads the DiT** (`gc.collect()` + `torch.cuda.empty_cache()` + `comfy.model_management.soft_empty_cache()`).
2. Only after VRAM is clean does it load the 5Hz LM, run PMI + reward scoring, and unload the LM in its `finally` block.
3. Interpreting `overall_score` and `pmi_details` (`report_json`):
   - Normalized PMI scores `> 0.5` indicate positive alignment between audio codes and prompt conditions (`0.7+` = strong adherence).
   - `dit_lyric_score` values closer to `1.0` indicate crisp, well-timed vocal articulation.
