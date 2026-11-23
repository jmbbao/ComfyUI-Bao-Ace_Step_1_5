# 12 GB VRAM Optimization Guide (XL 4B + 5Hz LM 4B)

**ComfyUI-Bao-Ace_Step_1_5** is specifically architected so consumer **12 GB VRAM GPUs** (e.g. RTX 3060 12 GB, RTX 4070 / 4070 Ti 12 GB) can run the largest ACE-Step 1.5 models (**DiT XL 4B + 5Hz LM 4B + AutoencoderOobleck VAE**) without Out-Of-Memory (OOM) errors.

---

## Peak VRAM Budget by Isolated Stage

Because loaders only return lazy handles and every execution node enforces `unload_after_run=True`, stages **never** overlap in VRAM:

| Stage | Active Components | BF16 Peak VRAM | INT8 Dynamic Peak VRAM | After Stage |
| --- | --- | :---: | :---: | :---: |
| 1. `AceStepVAEEncode` | `AutoencoderOobleck` (tiled 15 s) | ~1.0 GB | ~1.0 GB | ~0 GB |
| 2. `AceStepLMPlanner` / `QueryRewriter` / `Understanding` | `acestep-5Hz-lm-4B` (`pt` backend) | ~8.8 GB | **~4.6 GB** | ~0 GB |
| 3a. `AceStepSampler` (Text Conditioning) | `Qwen3-Embedding-0.6B` only | ~1.4 GB | ~1.4 GB | offloaded to CPU |
| 3b. `AceStepSampler` (Diffusion) | `acestep-v15-xl-*` (4B) | ~9.8 GB *(or ~3.5 GB w/ block offload)* | **~5.4 GB** | ~0 GB |
| 4. `AceStepVAEDecode` | `AutoencoderOobleck` (tiled 256 frames) | ~1.2 GB | ~1.2 GB | ~0 GB |

---

## Mandatory Best Practices for 12 GB GPUs

1. **Always keep `unload_after_run=True`:** Every LM, Sampler, VAE, and Analysis node frees GPU tensors and runs `gc.collect()`, `torch.cuda.empty_cache()`, and `comfy.model_management.soft_empty_cache()` inside a `try ... finally` block.
2. **Use `precision="int8_dynamic"` + disk caching (`save_quantized=True`):**
   - On your first run, enable `save_quantized=True` with `save_suffix="_int8_convrot"` on `AceStepDiTLoader` and `AceStepLMLoader`.
   - Only decoder `nn.Linear` layers are quantized (strictly excluding FSQ `tokenizer`, `detokenizer`, embeddings, and `lm_head` to preserve audio quality).
   - Quantized weights (`int8` + per-channel `float32` scales) are saved to `ComfyUI/models/acestep/<model_name>_int8_convrot/model_int8.safetensors` alongside `quant_meta.json`, `config.json`, `silence_latent.pt` (for DiT) or tokenizer files (for LM), and model code scripts.
   - **Keep only `*_int8_convrot` to save disk space:** Both loaders resolve the cached `*_int8_convrot` folder **before** checking or downloading the original checkpoint. You can safely delete the original full-precision folder manually (or enable `delete_original_after_quantize=True`, which verifies export completeness before deleting) and keep `auto_download=True` without ever re-downloading the original model.
   - Selecting a `*_int8_convrot` folder directly in `model_path` auto-detects INT8 even if `precision` is set to `bf16` or `fp16`. If `Qwen3-Embedding-0.6B` is missing, only `Qwen3-Embedding-0.6B/*` is downloaded (never pulling unrelated base models).
3. **Keep `offload_dit_to_cpu=True` and `vram_offload_threshold_gb=1.5`:**
   - If you run BF16 XL 4B or very long tracks (300–600 s with CFG), `DiTBlockOffloadManager` automatically activates block-by-block CPU streaming whenever projected free VRAM falls below `1.5 GB`, preventing OOM while keeping embeddings and projections fast on GPU.
4. **Use Tiled VAE Encoding & Decoding:**
   - `AceStepVAEEncode`: `use_tiled_encode=True`, `chunk_seconds=15.0`, `overlap_seconds=2.0`.
   - `AceStepVAEDecode`: `use_tiled_decode=True`, `chunk_size=256`, `overlap=64`, `offload_wav_to_cpu=True`, `fallback_to_cpu_on_oom=True`.
5. **Keep `batch_size=1` for XL 4B + LM 4B:** Increase `batch_size` to `2` only when using 2B DiT models or INT8 XL turbo on shorter tracks (<= 90 s).

---

## Python 3.13 & `torchao` Compatibility Notes

- **PyTorch `pt` backend:** The 5Hz LM always runs on the native PyTorch (`pt`) backend rather than `vllm`/`nano-vllm`, guaranteeing full compatibility with **Python 3.13**, Windows, and pre-Volta CUDA GPUs while permitting instantaneous VRAM release.
- **`torchao` version matching:** Pin `torchao` to match your PyTorch build (`torchao>=0.9.0,<0.18.0`; for PyTorch `2.7.x` use `torchao<=0.11.0` to avoid `ScalingType` import errors).
- **Automatic pure-PyTorch fallback:** If `torchao` is absent or incompatible with a specific PyTorch/Python 3.13 wheel, `memory_utils.py` seamlessly falls back to `Int8DynamicLinear` with identical `.safetensors` save/load support.
