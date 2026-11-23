"""ComfyUI-Bao-Ace_Step_1_5 — Complete Modular ACE-Step 1.5 Custom Nodes for ComfyUI.

Provides strictly isolated, sequential nodes engineered to run demanding
ACE-Step 1.5 workflows (4B XL DiT + 4B 5Hz LM + AutoencoderOobleck VAE) within
12 GB VRAM across Text-to-Music, Simple Mode, Query Rewriting, Metadata Control,
Reference Audio, Cover & Flow-Edit, Repaint & Edit, Track Separation (Extract),
Multi-Track Generation (Lego / Complete), Vocal2BGM, Audio Understanding,
LRC Generation, and Quality Scoring.
"""

from __future__ import annotations

from .memory_utils import get_acestep_models_dir
from .nodes_analysis import (
    NODE_CLASS_MAPPINGS as ANALYSIS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ANALYSIS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_lm import (
    NODE_CLASS_MAPPINGS as LM_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LM_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_loader import (
    NODE_CLASS_MAPPINGS as LOADER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LOADER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_sampler import (
    NODE_CLASS_MAPPINGS as SAMPLER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_tasks import (
    NODE_CLASS_MAPPINGS as TASKS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TASKS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_vae import (
    NODE_CLASS_MAPPINGS as VAE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VAE_NODE_DISPLAY_NAME_MAPPINGS,
)

# Register ComfyUI/models/acestep on import
get_acestep_models_dir()

NODE_CLASS_MAPPINGS = {
    **LOADER_NODE_CLASS_MAPPINGS,
    **LM_NODE_CLASS_MAPPINGS,
    **TASKS_NODE_CLASS_MAPPINGS,
    **SAMPLER_NODE_CLASS_MAPPINGS,
    **VAE_NODE_CLASS_MAPPINGS,
    **ANALYSIS_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOADER_NODE_DISPLAY_NAME_MAPPINGS,
    **LM_NODE_DISPLAY_NAME_MAPPINGS,
    **TASKS_NODE_DISPLAY_NAME_MAPPINGS,
    **SAMPLER_NODE_DISPLAY_NAME_MAPPINGS,
    **VAE_NODE_DISPLAY_NAME_MAPPINGS,
    **ANALYSIS_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
