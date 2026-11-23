# >> WORK IN PROGRESS YET, NOT FINISHED <<

# ComfyUI-Bao-Ace_Step_1_5

Modular, VRAM-isolated **ACE-Step 1.5** custom nodes for ComfyUI, engineered to run **DiT XL (4B) + 5Hz LM (4B) + AutoencoderOobleck VAE** reliably on **12 GB VRAM GPUs** on **Python 3.13** (and Python 3.12).

## Complete Feature Set (Phase 1 + Phase 2)

- **Core Text-to-Music & Simple Mode:** Natural-language query rewriting, structured lyrics (`[Verse]`, `[Chorus]`, etc.), 50+ vocal languages, instrumental mode, granular metadata control, and `10s–600s` duration.
- **5Hz Language Model (`pt` backend):** `acestep-5Hz-lm-0.6B`, `1.7B`, and `4B` for Chain-of-Thought planning, advanced Query Rewriting, 5Hz semantic codes, Audio Understanding, and LM PMI Quality Scoring.
- **DiT 2B & XL 4B Diffusion (`base` / `sft` / `turbo`):** 25Hz latent synthesis with ODE/SDE, Euler/Heun, CFG, ADG, dynamic block-level CPU offloading (`offload_dit_to_cpu`), and on-the-fly **INT8 dynamic quantization** with `.safetensors` disk caching (`_int8_convrot`).
- **Audio Conditioning & Editing Tasks:**
  - **Reference Audio Input** (`AceStepReferenceAudio`)
  - **Cover Generation & Flow-Edit Morph** (`AceStepCoverTask`)
  - **Repaint & Edit** with step injection + CPU waveform splice (`AceStepRepaintTask` + `AceStepVAEDecode`)
  - **Track Separation / Extract** (`AceStepTrackExtractTask`)
  - **Multi-Track Generation / Lego & Complete** (`AceStepMultiTrackTask`)
  - **Vocal2BGM** (`AceStepVocal2BGMTask`)
- **Analysis, Timestamps & Quality Evaluation:**
  - **Audio Tokenizer (5Hz Codes)** (`AceStepAudioTokenizer`)
  - **Audio Understanding** (`AceStepAudioUnderstanding`)
  - **LRC Generation** with DTW cross-attention alignment and optional `.lrc` file export (`AceStepLRCGenerator`)
  - **Quality Scoring** combining DiT lyric alignment + sequential 5Hz LM PMI/reward scoring (`AceStepQualityScorer`)

## Installation (WARNING: not finished yet, commented out entries in requirements.txt)

1. Place this folder in `ComfyUI/custom_nodes/ComfyUI-Bao-Ace_Step_1_5/`.
2. Install dependencies into ComfyUI's Python environment:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-Bao-Ace_Step_1_5/requirements.txt
```

3. Models are stored in `ComfyUI/models/acestep/`. With `auto_download=True` (default), official Hugging Face checkpoints (`ACE-Step/Ace-Step1.5` and sub-models) are fetched automatically on first use.

## Node Catalog (19 Nodes)

| Category | Nodes |
| --- | --- |
| `ACE-Step 1.5/Loaders` | `AceStepLMLoader`, `AceStepDiTLoader`, `AceStepVAELoader` |
| `ACE-Step 1.5/VAE` | `AceStepVAEEncode`, `AceStepVAEDecode` |
| `ACE-Step 1.5/LM` | `AceStepMetadataControl`, `AceStepQueryRewriter`, `AceStepLMPlanner` |
| `ACE-Step 1.5/Tasks` | `AceStepReferenceAudio`, `AceStepCoverTask`, `AceStepRepaintTask`, `AceStepTrackExtractTask`, `AceStepMultiTrackTask`, `AceStepVocal2BGMTask` |
| `ACE-Step 1.5/Sampling` | `AceStepSampler` |
| `ACE-Step 1.5/Analysis` | `AceStepAudioTokenizer`, `AceStepAudioUnderstanding`, `AceStepLRCGenerator`, `AceStepQualityScorer` |

## Documentation (`docs/`)

- [Node Reference](docs/NODES_REFERENCE.md) — Complete inputs, outputs, types, defaults, and 12 GB recommendations for all 19 nodes.
- [Tasks, Editing & Metadata Guide](docs/TASKS_AND_EDITING.md) — Reference Audio, Cover/Flow-Edit, Repaint & Splice, Track Separation, Multi-Track, Vocal2BGM, and checkpoint compatibility matrix.
- [Analysis, LRC & Quality Scoring Guide](docs/ANALYSIS_AND_SCORING.md) — Audio Understanding, LRC `.lrc` timestamp generation, and sequential DiT + LM PMI quality scoring.
- [12 GB VRAM Optimization Guide](docs/VRAM_12GB_GUIDE.md) — Peak VRAM breakdown, INT8 `.safetensors` caching (`_int8_convrot`), block offloading, and Python 3.13 / `torchao` tips.

## Ready-to-Load Workflows (`workflows/`)

Drag and drop any `.json` file from `workflows/` directly into ComfyUI:

1. `workflows/01_reference_audio_input.json` — Style/timbre transfer from reference audio
2. `workflows/02_cover_generation.json` — Cover & Remix (with optional Flow-Edit morph)
3. `workflows/03_repaint_and_edit.json` — Temporal Repaint & Edit with CPU waveform splice
4. `workflows/04_track_separation.json` — Track Separation (Extract vocals/drums/bass/etc., `xl-base`)
5. `workflows/05_multi_track_generation.json` — Multi-Track Generation (Lego / Complete, `xl-base`)
6. `workflows/06_vocal2bgm.json` — Full BGM accompaniment from a vocal track (`xl-base`)
7. `workflows/07_metadata_control.json` — Granular musical metadata & structured prompt composition
8. `workflows/08_simple_mode.json` — Simple Mode end-to-end (XL 4B + LM 4B in 12 GB VRAM)
9. `workflows/09_query_rewriting.json` — Advanced 5Hz LM Query Rewriting into synthesis
10. `workflows/10_audio_understanding.json` — Reverse-engineer caption, lyrics, and metadata from audio
11. `workflows/11_lrc_generation.json` — Generate music + aligned `.lrc` timestamps
12. `workflows/12_quality_scoring.json` — Generate music + sequential DiT alignment & LM PMI scoring
