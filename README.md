# >> WORK IN PROGRESS << 
## >> NOT FINISHED BUT ALREADY WORKING <<

# ComfyUI-Bao-Ace_Step_1_5

Nodes for running Ace Step 1.5 / 1.5 XL on a graphics card with 12 GB of VRAM or more.
Enables the execution —using only 12 GB of VRAM— of configurations that previously required 24 GB of VRAM (LM 4B + Ace Step 1.5XL), achieved through unloading the LM from VRAM and the use of int8_convrot quantization. When downloading a model it will create a int8_convrot version if you turn on that option, it can delete the original leaving only the int8_convrot quantizations if you check the delete original option (I recommend first to test qualitiy on the original versus the int8_convrot before deleting)

Modular, VRAM-isolated **ACE-Step 1.5** custom nodes for ComfyUI, engineered to run **DiT XL (4B) + 5Hz LM (4B) + AutoencoderOobleck VAE** reliably on **12 GB VRAM GPUs** on **Python 3.13** (and Python 3.12).

## Complete Feature Set

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

## INSTALLATION

I will use the /ComfyUI folder, if your ComfyUI folder is named differently pay attention to change it in the orders below.

1. Dowload the nodes ComfyUI-Bao-Ace_Step_1_5

Open a terminal in /ComfyUI/custom_nodes:
```bash
git clone https://github.com/jmbbao/ComfyUI-Bao-Ace_Step_1_5.git
```

The folder /ComfyUI/custom_nodes/ComfyUI-Bao-Ace_Step_1_5 will be created.


2. Install dependencies into ComfyUI's Python environment:

For Linux with venv environment: 
(for windows or a different environment, ask some AI to convert these orders to the ones you need, it will ask you to know if you uses uv environment, etc)

Open a terminal in /ComfyUI folder (root of ComfyUI) and (in Linux with venv) execute these orders:
```bash
source venv/bin/activate
pip install -r ./custom_nodes/ComfyUI-Bao-Ace_Step_1_5/requirements.txt
deactivate
```


3. I have disabled one of the entries in requirements.txt because it could change things in the system and it is better to install it manually and also because you have to make a change if you have Python 3.13 because by default it only allows Python 3.12

The package appears like this in requirements.txt: (it's a comment for that initial #, it hasn't been installed)
```bash
# git+https://github.com/ace-step/ACE-Step-1.5.git
```

We install it like this:

Open a terminal in the /ComfyUI directory and download the code (it will go down to /ComfyUI/ACE-Step-1.5):
```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
```

Now, in the file /ComfyUI/ACE-Step-1.5/pyproject.toml, change
the line that reads:

requires-python = ">=3.11,<3.13"

so that it says:

requires-python = ">=3.11,<3.14"

It will also work with Python 3.13, in addition to 3.12.

In `pyproject.toml`, look for the actual package name; it appears after `[project]`.
You will see the following:
```bash
[project]
name = "ace-step"
```
Once installed, the actual name will be: ace-step (we will use this when uninstalling).

We install it using pip, without including any of its dependencies, to ensure that no other packages are altered:
```bash
pip install --no-deps ./ACE-Step-1.5
```

You can now delete the /ComfyUI/ACE-Step-1.5 folder as it is no longer used.

When you want to uninstall the package, indicate its real name with which it was installed:
```bash
pip uninstall ace-step
```

You can check if something is installed with:

pip show ace-step


4. If you are using `extra_model_paths.yaml` to point to a `/models` folder other than `/ComfyUI/models`, the models will be downloaded to the `acestep` folder that you specify in `extra_model_paths.yaml`—make sure to include it.

Example `extra_model_paths.yaml` entry:
```yaml
shared_models:
  base_path: /path/to/your/models
  acestep: acestep
  ...
```

With `auto_download=True` (default), official Hugging Face checkpoints (`ACE-Step/Ace-Step1.5` and sub-models) are fetched automatically into that folder on first use.

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
