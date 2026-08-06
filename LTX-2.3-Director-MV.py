# ======================================================================
# LTX 2.3 Director 2.0 -- Music Video Pipeline (30s @ 24fps)
#
# Architecture:
#   CELL 1   Environment Setup (subprocess + Colab magic dual-pattern)
#   CELL 2   Model Downloads (aria2c)
#   CELL 3   Core Imports & Utilities
#   CELL 4   Model Loading (UNet, CLIP, LoRAs, VAEs, Upscaler)
#   CELL 5   LTXDirector Setup (timeline, audio, prompts)
#   CELL 6   Conditioning & Pass 1 Pipeline
#   CELL 7   Pass 2 Pipeline (Upscale + Refine)
#   CELL 8   Decode & Output (Video + Audio combine)
#   CELL 9   Configuration & Run
#
# Workflow source: "1. LTX 2.3_Director_2.0 MV Workflow 30s 260802-2.json"
# Total nodes: 32
# GPU target: Google Colab T4 (15 GB) / L4 (24 GB)
# Model: LTX 2.3 22B Q4_K_M GGUF (Kijai)
# Pipeline: Two-pass Director with 4 LoRAs + Audio + Timeline
#   Pass 1: 8 steps, euler, linear_quadratic, denoise=1.0, strength=0.5
#   Pass 2: 4 steps, euler, linear_quadratic, denoise=0.42, strength=1.0
# Resolution: 1280x720 @ 24fps, 756 frames (31.5 seconds)
# Output: h264-mp4, CRF=8, yuv420p
#
# Node mapping (JSON node ID -> Python section):
#   135: UnetLoaderGGUF          -> CELL 4
#   12:  DualCLIPLoader          -> CELL 4
#   138: Power Lora Loader       -> CELL 4
#   6:   VAELoaderKJ (tiny)      -> CELL 4
#   8:   VAELoader (audio)       -> CELL 4
#   36:  VAELoader (video)       -> CELL 4
#   13:  LatentUpscaleModelLoader -> CELL 4
#   10:  ModelPreviewOverrideKJ  -> CELL 4
#   131: LTXDirector             -> CELL 5
#   128: ConditioningZeroOut     -> CELL 6
#   27:  LTXVConditioning        -> CELL 6
#   133: LTXDirectorGuide (P1)   -> CELL 6
#   29:  LTXVConcatAVLatent (P1) -> CELL 6
#   28:  CFGGuider (P1)          -> CELL 6
#   32:  KSamplerSelect (P1)     -> CELL 6
#   33:  BasicScheduler (P1)     -> CELL 6
#   30:  RandomNoise             -> CELL 6
#   31:  SamplerCustomAdvanced(P1)-> CELL 6
#   34:  LTXVSeparateAVLatent    -> CELL 7
#   55:  LTXDirectorCropGuides   -> CELL 7
#   14:  LTXVLatentUpsampler     -> CELL 7
#   132: LTXDirectorGuide (P2)   -> CELL 7
#   18:  LTXVConcatAVLatent (P2) -> CELL 7
#   17:  CFGGuider (P2)          -> CELL 7
#   20:  KSamplerSelect (P2)     -> CELL 7
#   21:  BasicScheduler (P2)     -> CELL 7
#   19:  SamplerCustomAdvanced(P2)-> CELL 7
#   22:  LTXVSeparateAVLatent    -> CELL 8
#   54:  LTXDirectorCropGuides   -> CELL 8
#   1:   VAEDecode               -> CELL 8
#   24:  LTXVAudioVAEDecode      -> CELL 8
#   139: VHS_VideoCombine        -> CELL 8
# ======================================================================


# ======================================================================
# CELL 1 -- ENVIRONMENT SETUP (run once per Colab session)
# ======================================================================
# @title { "single-column": true }
# @markdown ## 1. Install Environment

import subprocess
import os
import sys


def _shell(cmd):
    """Run a shell command (replaces ! magic for AST compatibility)."""
    subprocess.run(cmd, shell=True, check=True)


def _shell_safe(cmd):
    """Run a shell command but only warn on failure instead of raising.

    Use for non-critical operations (custom node clones, optional pip installs)
    so that a missing repo or transient failure does not halt the entire setup.
    """
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Command failed (non-fatal): {cmd}\n  -> {e}")


def _pip(packages):
    """pip install helper."""
    _shell(f"pip install -q {packages}")


# -- Base Python packages (subprocess version for non-Colab execution) --
_shell("pip install torch torchvision torchaudio")

# -- Colab magic equivalent (dual-pattern) --
# !pip install torch torchvision torchaudio

# %cd /content
# from IPython.display import clear_output
# clear_output()

# !pip install -q torchsde einops diffusers accelerate nest_asyncio
# !pip install -q av spandrel albumentations onnx opencv-python onnxruntime
# !pip install -q imageio imageio-ffmpeg
# !pip install -q transformers>=4.43.0 accelerate huggingface_hub

_pip("torchsde einops diffusers accelerate nest_asyncio")
_pip("av spandrel albumentations onnx opencv-python onnxruntime")
_pip("imageio imageio-ffmpeg")
_pip("transformers>=4.43.0 accelerate huggingface_hub")

# -- ComfyUI (official repo, no branch pin for stability) --
# !git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI
if not os.path.exists("/content/ComfyUI"):
    _shell("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
_shell("pip install -r /content/ComfyUI/requirements.txt -q")

# -- Custom nodes --
# %cd /content/ComfyUI/custom_nodes
CUSTOM_NODES = "/content/ComfyUI/custom_nodes"
os.makedirs(CUSTOM_NODES, exist_ok=True)

# Core KJNodes (VAELoaderKJ, ModelPreviewOverrideKJ, etc.)
# !git clone https://github.com/kijai/ComfyUI-KJNodes.git
if not os.path.exists(f"{CUSTOM_NODES}/ComfyUI-KJNodes"):
    _shell_safe(f"git clone https://github.com/kijai/ComfyUI-KJNodes.git {CUSTOM_NODES}/ComfyUI-KJNodes")

# GGUF loader (UnetLoaderGGUF)
# !git clone https://github.com/city96/ComfyUI-GGUF.git
if not os.path.exists(f"{CUSTOM_NODES}/ComfyUI-GGUF"):
    _shell_safe(f"git clone https://github.com/city96/ComfyUI-GGUF.git {CUSTOM_NODES}/ComfyUI-GGUF")

# LTXVideo nodes (LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent, etc.)
# !git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
if not os.path.exists(f"{CUSTOM_NODES}/ComfyUI-LTXVideo"):
    _shell_safe(f"git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git {CUSTOM_NODES}/ComfyUI-LTXVideo")

# VideoHelperSuite (VHS_VideoCombine)
# !git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
if not os.path.exists(f"{CUSTOM_NODES}/ComfyUI-VideoHelperSuite"):
    _shell_safe(f"git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git {CUSTOM_NODES}/ComfyUI-VideoHelperSuite")

# rgthree-comfy (Power Lora Loader)
# !git clone https://github.com/rgthree/rgthree-comfy.git
if not os.path.exists(f"{CUSTOM_NODES}/rgthree-comfy"):
    _shell_safe(f"git clone https://github.com/rgthree/rgthree-comfy.git {CUSTOM_NODES}/rgthree-comfy")

# whatdreamscost-comfyui (LTXDirector, LTXDirectorGuide, LTXDirectorCropGuides)
# !git clone https://github.com/whatdreamscost/whatdreamscost-comfyui.git
if not os.path.exists(f"{CUSTOM_NODES}/whatdreamscost-comfyui"):
    _shell_safe(f"git clone https://github.com/whatdreamscost/whatdreamscost-comfyui.git {CUSTOM_NODES}/whatdreamscost-comfyui")

# -- Install custom node requirements --
for node_dir in ["ComfyUI-KJNodes", "ComfyUI-GGUF", "ComfyUI-LTXVideo",
                 "ComfyUI-VideoHelperSuite", "rgthree-comfy", "whatdreamscost-comfyui"]:
    req_path = f"{CUSTOM_NODES}/{node_dir}/requirements.txt"
    if os.path.exists(req_path):
        _shell_safe(f"pip install -r {req_path} -q")

# -- System packages --
subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

# -- Final path setup --
os.chdir("/content/ComfyUI")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")
print("Environment ready. All custom nodes installed.")



# ======================================================================
# CELL 2 -- MODEL DOWNLOADS (run once; skips cached files)
# ======================================================================
# @title { "single-column": true }
# @markdown ## 2. Download Model Weights

import os
import subprocess
from pathlib import Path


def _dl(url: str, dest: str, filename: str = None) -> str:
    """Download a file with aria2c, skip if cached and >1MB.

    Captures both stdout and stderr from aria2c for improved error reporting.
    When aria2c returns non-zero (e.g. HTTP 404), stderr may be empty so we
    also inspect stdout for diagnostic information.
    """
    Path(dest).mkdir(parents=True, exist_ok=True)
    fn = filename or url.split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  cached  {fn}")
        return fn
    cmd = [
        "aria2c", "--console-log-level=error", "--summary-interval=0", "--quiet",
        "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", dest, "-o", fn, url
    ]
    print(f"  dl {fn}...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Combine stderr and stdout for diagnostics -- aria2c may report
        # errors in either stream depending on the failure mode (404 vs
        # network timeout vs DNS failure).
        err_detail = (r.stderr.strip() or r.stdout.strip() or
                      f"aria2c exit code {r.returncode}")
        raise RuntimeError(
            f"Download failed for {fn} from {url}: {err_detail}. "
            f"Check network connection and re-run Cell 2.")
    print("done.")
    return fn


def _dl_multi(urls: list, dest: str, filename: str = None) -> str:
    """Try downloading from multiple URLs in order (fallback mechanism).

    Accepts a list of URLs and attempts each one sequentially. Returns the
    filename on first success. If all URLs fail, raises RuntimeError with
    details of every attempt.

    Args:
        urls: List of full download URLs to try in priority order.
        dest: Destination directory path.
        filename: Optional override filename. If None, derived from last URL
                  path component.

    Returns:
        The downloaded filename (basename only).
    """
    fn = filename or urls[0].split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    # Skip download if already cached
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  cached  {fn}")
        return fn

    errors = []
    for i, url in enumerate(urls, 1):
        try:
            result = _dl(url, dest, filename=fn)
            return result
        except RuntimeError as e:
            errors.append(f"  Attempt {i}/{len(urls)} ({url}): {e}")
            # Clean up any partial/empty file before trying next URL
            if os.path.exists(full) and os.path.getsize(full) < 1_000_000:
                os.remove(full)
            continue

    # All URLs failed
    raise RuntimeError(
        f"All download sources failed for {fn}:\n" + "\n".join(errors) + "\n"
        f"Please verify HuggingFace repo availability and re-run Cell 2."
    )


# -- Download directories --
KIJAI_GGUF = "https://huggingface.co/Kijai/LTX-Video-2.3-GGUF/resolve/main"
CITY96_GGUF = "https://huggingface.co/city96/LTX-Video-2.3-gguf/resolve/main"
CORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LTX_LIGHT = "https://huggingface.co/Lightricks/LTX-Video-2.3/resolve/main"
LTX_HF = "https://huggingface.co/Lightricks"
UNET_D = "/content/ComfyUI/models/unet"
TE_D = "/content/ComfyUI/models/text_encoders"
VAE_D = "/content/ComfyUI/models/vae"
UPD = "/content/ComfyUI/models/latent_upscale_models"
LR_D = "/content/ComfyUI/models/loras"

# ============================================================
# Core Models (Nodes 135, 12, 6, 8, 36, 13)
# ============================================================
print("-- Core Models --")

# Node 135: UnetLoaderGGUF - ltx-2-3-22b-dev-Q4_K_M.gguf
_M_UNET = _dl_multi([
    f"{CITY96_GGUF}/ltx-2-3-22b-dev-Q4_K_M.gguf",
    f"{KIJAI_GGUF}/ltx-2-3-22b-dev-Q4_K_M.gguf",
], UNET_D)

# Node 12: DualCLIPLoader - gemma_3_12B + ltx-2.3 text projection
_M_CLIP1 = _dl(
    f"{CORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", TE_D
)
_M_CLIP2 = _dl_multi([
    f"{KIJAI_GGUF}/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    f"{LTX_LIGHT}/ltx-2.3_text_projection_bf16.safetensors",
], TE_D)

# Node 36: VAELoader - LTX23_video_vae_bf16.safetensors
_M_VVAE = _dl_multi([
    f"{KIJAI_GGUF}/vae/LTX23_video_vae_bf16.safetensors",
    f"{LTX_LIGHT}/LTX23_video_vae_bf16.safetensors",
], VAE_D)

# Node 8: VAELoader - LTX23_audio_vae_bf16.safetensors
_M_AVAE = _dl_multi([
    f"{KIJAI_GGUF}/vae/LTX23_audio_vae_bf16.safetensors",
    f"{LTX_LIGHT}/LTX23_audio_vae_bf16.safetensors",
], VAE_D)

# Node 6: VAELoaderKJ - taeltx2_3.safetensors (tiny preview VAE)
_M_TAEV = _dl_multi([
    f"{KIJAI_GGUF}/vae/taeltx2_3.safetensors",
    "https://huggingface.co/madebyollin/taeltxv/resolve/main/taeltx2_3.safetensors",
], VAE_D)

# Node 13: LatentUpscaleModelLoader - ltx-2.3-spatial-upscaler-x2-1.1
_M_UP = _dl_multi([
    f"{LTX_HF}/LTX-Video/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    f"{LTX_HF}/LTX-2/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
], UPD)

# ============================================================
# LoRAs (Node 138: Power Lora Loader - 4 LoRAs)
# ============================================================
print("\n-- LoRAs --")

# LoRA 1: ltx-2.3-22b-distilled-lora-dynamic @ 0.4
_dl_multi([
    f"{KIJAI_GGUF}/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    f"{LTX_HF}/LTX-Video-2.3-loras/resolve/main/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
], LR_D)
# LoRA 2: LTX-2.3-OmniNFT-RL-Lora_bf16 @ 0.6
_dl_multi([
    f"{KIJAI_GGUF}/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    f"{LTX_HF}/LTX-Video-2.3-loras/resolve/main/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
], LR_D)
# LoRA 3: ltx2.3-transition @ 0.7
_dl_multi([
    f"{KIJAI_GGUF}/loras/ltx2.3-transition.safetensors",
    f"{LTX_HF}/LTX-Video-2.3-loras/resolve/main/ltx2.3-transition.safetensors",
], LR_D)
# LoRA 4: LTX2.3-MVCamera-drclips @ 0.9
_dl_multi([
    f"{KIJAI_GGUF}/loras/LTX2.3-MVCamera-drclips.safetensors",
    f"{LTX_HF}/LTX-Video-2.3-loras/resolve/main/LTX2.3-MVCamera-drclips.safetensors",
], LR_D)

print("\nAll models ready.")



# ======================================================================
# CELL 3 -- CORE IMPORTS & UTILITIES
# ======================================================================
# @title { "single-column": true }
# @markdown ## 3. Imports, Helpers, Node Loader

import gc
import json
import time
import warnings
import asyncio
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping
from IPython.display import display, HTML, clear_output

warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/ComfyUI")
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths


# ======================================================================
# VRAM Management Helpers
# ======================================================================

def _vram_free():
    """Release GPU memory: gc + empty CUDA cache + ipc_collect."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def _vram_print(tag=""):
    """Print a visual VRAM usage bar."""
    if not torch.cuda.is_available():
        return
    u = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * u / t) if t else 0
    bar = "|" * filled + "." * (20 - filled)
    print(f"   VRAM [{bar}] {u:.1f}/{t:.1f} GB  {tag}")


# ======================================================================
# Tensor / Node Output Helpers
# ======================================================================

def get_value_at_index(obj: Union[Sequence, Mapping], idx: int) -> Any:
    """Extract a value from ComfyUI node output (list or dict)."""
    try:
        return obj[idx]
    except KeyError:
        return obj["result"][idx]


# ======================================================================
# ComfyUI Node Loader
# ======================================================================

_NODES_LOADED = False


def _load_comfy_nodes():
    """Load ComfyUI built-in and external custom nodes (async-safe)."""
    global _NODES_LOADED
    if _NODES_LOADED:
        return
    import nest_asyncio
    nest_asyncio.apply()
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def _l():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print(f"   [Nodes] Some nodes failed to load (non-critical): {len(failed)} failures")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(_l())
        else:
            asyncio.run(_l())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_l())

    # Verify critical nodes for this workflow are available
    critical_nodes = [
        "UnetLoaderGGUF", "DualCLIPLoader", "Power Lora Loader (rgthree)",
        "VAELoaderKJ", "VAELoader", "LatentUpscaleModelLoader",
        "ModelPreviewOverrideKJ", "LTXDirector", "LTXDirectorGuide",
        "LTXDirectorCropGuides", "ConditioningZeroOut", "LTXVConditioning",
        "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVLatentUpsampler",
        "LTXVAudioVAEDecode", "CFGGuider", "KSamplerSelect", "BasicScheduler",
        "RandomNoise", "SamplerCustomAdvanced", "VAEDecode", "VHS_VideoCombine",
    ]
    missing = [n for n in critical_nodes if n not in NODE_CLASS_MAPPINGS]
    if missing:
        print(f"   [Nodes] WARNING: Missing critical nodes: {missing}")
        print(f"   [Nodes] Available: {len(NODE_CLASS_MAPPINGS)} nodes loaded")
    else:
        print(f"   [Nodes] All {len(critical_nodes)} critical nodes available")

    _NODES_LOADED = True


_load_comfy_nodes()
print("Core utilities ready.")



# ======================================================================
# CELL 4 -- MODEL LOADING
# ======================================================================
# @title { "single-column": true }
# @markdown ## 4. Load Models (UNet, CLIP, LoRAs, VAEs, Upscaler)
# Strategy: load -> use -> unload in sequence to manage VRAM

print("=" * 60)
print("CELL 4: Loading models...")
print("=" * 60)

_vram_free()

# ============================================================
# Node 135: UnetLoaderGGUF
# Loads: ltx-2-3-22b-dev-Q4_K_M.gguf
# Output: MODEL -> feeds into Power Lora Loader (node 138)
# ============================================================
print("\n[1/8] Loading UNet (GGUF Q4_K_M 22B)...")
_vram_print("before UNet")

unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
unetloadergguf_result = unetloadergguf.load_unet(
    unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"
)
unet_model = get_value_at_index(unetloadergguf_result, 0)

_vram_print("after UNet")
print("   UNet loaded.")

# ============================================================
# Node 12: DualCLIPLoader
# Loads: gemma_3_12B_it_fp4_mixed.safetensors (CLIP1)
#        ltx-2.3_text_projection_bf16.safetensors (CLIP2)
# Type: ltxv
# Output: CLIP -> feeds into Power Lora Loader (node 138)
# ============================================================
print("\n[2/8] Loading Dual CLIP (Gemma 3 12B + LTX text projection)...")

dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
dualcliploader_result = dualcliploader.load_clip(
    clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
    clip_name2="ltx-2.3_text_projection_bf16.safetensors",
    type="ltxv",
    device="default"
)
clip_model = get_value_at_index(dualcliploader_result, 0)

_vram_print("after CLIP")
print("   Dual CLIP loaded.")

# ============================================================
# Node 138: Power Lora Loader (rgthree)
# Applies 4 LoRAs to model + clip:
#   1. ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16 @ 0.4
#   2. LTX-2.3-OmniNFT-RL-Lora_bf16 @ 0.6
#   3. ltx2.3-transition @ 0.7
#   4. LTX2.3-MVCamera-drclips @ 0.9
# Input: model (from node 135), clip (from node 12)
# Output: MODEL -> node 10 (ModelPreviewOverrideKJ)
#         CLIP -> node 131 (LTXDirector)
# ============================================================
print("\n[3/8] Applying 4 LoRAs via Power Lora Loader...")

# Define the LoRA stack configuration matching JSON widgets_values
LORA_STACK = [
    {"on": True, "lora": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
     "strength": 0.4, "strengthTwo": None},
    {"on": True, "lora": "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
     "strength": 0.6, "strengthTwo": None},
    {"on": True, "lora": "ltx2.3-transition.safetensors",
     "strength": 0.7, "strengthTwo": None},
    {"on": True, "lora": "LTX2.3-MVCamera-drclips.safetensors",
     "strength": 0.9, "strengthTwo": None},
]

# Try Power Lora Loader (rgthree) first, fallback to manual LoraLoaderModelOnly
if "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS:
    power_lora_loader = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()
    # The Power Lora Loader takes model, clip, and lora config widgets
    power_lora_result = getattr(power_lora_loader, power_lora_loader.FUNCTION)(
        model=unet_model,
        clip=clip_model,
        **{f"lora_{i+1}": lora for i, lora in enumerate(LORA_STACK)}
    )
    lora_model = get_value_at_index(power_lora_result, 0)
    lora_clip = get_value_at_index(power_lora_result, 1)
    print("   Power Lora Loader (rgthree) applied 4 LoRAs.")
else:
    # Fallback: apply LoRAs manually one by one
    print("   Power Lora Loader not found, using manual LoRA application...")
    lora_model = unet_model
    lora_clip = clip_model
    for slot in LORA_STACK:
        if slot["on"] and slot["lora"] not in (None, "None", ""):
            try:
                lora_result = LoraLoaderModelOnly().load_lora_model_only(
                    model=lora_model,
                    lora_name=slot["lora"],
                    strength_model=slot["strength"]
                )
                lora_model = get_value_at_index(lora_result, 0)
                print(f"      Applied: {slot['lora']} @ {slot['strength']}")
            except Exception as e:
                print(f"      SKIP: {slot['lora']}: {e}")

_vram_print("after LoRAs")

# ============================================================
# Node 6: VAELoaderKJ (Tiny Preview VAE)
# Loads: taeltx2_3.safetensors
# Device: main_device, Weight dtype: bf16
# Output: VAE -> node 10 (ModelPreviewOverrideKJ)
# ============================================================
print("\n[4/8] Loading Tiny Preview VAE (taeltx2_3)...")

vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
vaeloaderkj_result = vaeloaderkj.load_vae(
    vae_name="taeltx2_3.safetensors",
    device="main_device",
    weight_dtype="bf16"
)
tiny_vae = get_value_at_index(vaeloaderkj_result, 0)
print("   Tiny VAE loaded (bf16).")

# ============================================================
# Node 8: VAELoader (Audio VAE)
# Loads: LTX23_audio_vae_bf16.safetensors
# Output: VAE -> node 131 (LTXDirector, audio_vae input)
#              -> node 24 (LTXVAudioVAEDecode)
# ============================================================
print("\n[5/8] Loading Audio VAE...")

vaeloader_audio = NODE_CLASS_MAPPINGS["VAELoader"]()
vaeloader_audio_result = vaeloader_audio.load_vae(
    vae_name="LTX23_audio_vae_bf16.safetensors"
)
audio_vae = get_value_at_index(vaeloader_audio_result, 0)
print("   Audio VAE loaded.")

# ============================================================
# Node 36: VAELoader (Video VAE)
# Loads: LTX23_video_vae_bf16.safetensors
# Output: VAE -> node 14 (LTXVLatentUpsampler)
#              -> node 132 (LTXDirectorGuide Pass 2, vae)
#              -> node 133 (LTXDirectorGuide Pass 1, vae)
#              -> node 1 (VAEDecode)
# ============================================================
print("\n[6/8] Loading Video VAE...")

vaeloader_video = NODE_CLASS_MAPPINGS["VAELoader"]()
vaeloader_video_result = vaeloader_video.load_vae(
    vae_name="LTX23_video_vae_bf16.safetensors"
)
video_vae = get_value_at_index(vaeloader_video_result, 0)
print("   Video VAE loaded.")

# ============================================================
# Node 13: LatentUpscaleModelLoader
# Loads: ltx-2.3-spatial-upscaler-x2-1.1.safetensors
# Output: LATENT_UPSCALE_MODEL -> node 14 (LTXVLatentUpsampler)
# ============================================================
print("\n[7/8] Loading Latent Upscale Model (2x spatial)...")

latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
latentupscalemodelloader_result = latentupscalemodelloader.load_model(
    model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
upscale_model = get_value_at_index(latentupscalemodelloader_result, 0)
print("   Upscale model loaded.")

# ============================================================
# Node 10: ModelPreviewOverrideKJ
# Connects: model (from LoRA loader node 138) + tiny VAE (node 6)
# Widgets: [0, 80, true, 240, 24, ""]
# Output: MODEL -> node 131 (LTXDirector, model input)
# ============================================================
print("\n[8/8] Setting up Model Preview Override...")

modelpreviewoverridekj = NODE_CLASS_MAPPINGS["ModelPreviewOverrideKJ"]()
modelpreviewoverridekj_result = modelpreviewoverridekj.patch(
    model=lora_model,
    vae=tiny_vae,
    start=0,
    every_nth=80,
    limit_images=True,
    max_resolution=240,
    fps=24,
    prefix=""
)
preview_model = get_value_at_index(modelpreviewoverridekj_result, 0)
print("   Model preview override configured.")

_vram_free()
_vram_print("after all model loading")
print("\nAll models loaded successfully.")




# ======================================================================
# CELL 5 -- LTXDirector SETUP
# ======================================================================
# @title { "single-column": true }
# @markdown ## 5. LTXDirector Node (Timeline, Audio, Prompts)
# Node 131: LTXDirector (whatdreamscost-comfyui ver 2.0.0)
# This is the main orchestration node that creates:
#   - model (with director conditioning applied)
#   - positive conditioning (from global_prompt + timeline)
#   - video_latent (empty latent sized for 756 frames @ 1280x720)
#   - audio_latent (from audio VAE encoding of timeline audio)
#   - guide_data (image guide data from timeline segments)
#   - motion_guide_data (motion tracking data)
#   - frame_rate (24.0 float output)

print("=" * 60)
print("CELL 5: LTXDirector Setup (31.5s, 756 frames, 24fps)")
print("=" * 60)

_vram_free()

# -- Global Prompt (full text from JSON workflow node 131 properties) --
# This is the complete ~4000-word music video performance prompt
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
\u2022 Perform with explosive stage presence.
\u2022 Every musical phrase immediately creates a new emotional and physical performance.
\u2022 Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, hands, posture and body rhythm.
\u2022 The performance continuously builds toward emotional peaks.
\u2022 Own the stage with absolute confidence.
\u2022 Perform as if in front of 50,000 screaming fans.
\u2022 Captivate the audience every second.
\u2022 Never appear calm, passive or static.

Facial Performance:
\u2022 Extremely expressive facial acting throughout the entire performance.
\u2022 Rich emotional transitions every few words.
\u2022 Powerful eye contact with intense emotional engagement.
\u2022 Eyes sparkle with confidence and passion.
\u2022 Highly expressive eyebrows synchronized with important lyrics.
\u2022 Strong cheek and jaw movement while singing.
\u2022 Natural smiles, smirks, determination, excitement, confidence, attitude, passion, curiosity, joy and intensity.
\u2022 Rich cinematic micro-expressions.
\u2022 Never hold the same facial expression for more than a brief musical phrase.
\u2022 The face should feel emotionally alive every second.

Body Performance:
\u2022 The entire body constantly grooves with the beat.
\u2022 Strong rhythmic bouncing.
\u2022 Powerful shoulder accents.
\u2022 Confident chest movement.
\u2022 Hip movement follows the groove.
\u2022 Frequent body turns.
\u2022 Fast weight shifts.
\u2022 Dynamic torso twists.
\u2022 Lean toward the camera during emotional lyrics.
\u2022 Occasionally step toward the camera.
\u2022 Performance intensity increases naturally during powerful musical moments.
\u2022 Bold, energetic and theatrical stage movement.

Hand Performance:
\u2022 Perform like an experienced pop or hip-hop superstar.
\u2022 Large expressive gestures.
\u2022 Fast rhythmic arm accents.
\u2022 Sharp hand movements synchronized with the beat.
\u2022 Powerful pointing.
\u2022 Sweeping arm movements.
\u2022 Punching the air.
\u2022 Pulling gestures toward the chest.
\u2022 Throwing gestures outward.
\u2022 Finger snapping.
\u2022 Open palm emphasis.
\u2022 Framing the face.
\u2022 Expressive wrist movement.
\u2022 Hands constantly create visual rhythm.
\u2022 One hand naturally leads while the other follows.
\u2022 Asymmetrical movement.
\u2022 Avoid symmetrical gestures.
\u2022 Never repeatedly raise both hands together.
\u2022 Every musical phrase introduces fresh gestures.
\u2022 Never repeat the same gesture pattern.

Musical Timing:
\u2022 Body movement follows musical phrasing rather than every word.
\u2022 Strong beats create explosive movements.
\u2022 Soft phrases become intimate and emotional.
\u2022 Fast lyrics generate faster gestures.
\u2022 Slow lyrics become smoother without losing energy.
\u2022 Every movement feels rhythmically connected to the music.

Speech Synchronization:
\u2022 Perfect lip synchronization.
\u2022 Accurate mouth shapes.
\u2022 Expressions and gestures match the emotional meaning of every lyric.
\u2022 Natural breathing between phrases.

Motion Quality:
\u2022 Premium AI human animation.
\u2022 Fast, confident and energetic performance.
\u2022 Realistic momentum.
\u2022 Strong acceleration and deceleration.
\u2022 High-energy body mechanics.
\u2022 Natural motion blur.
\u2022 No robotic movement.
\u2022 No frozen poses.
\u2022 No repetitive gesture loops.
\u2022 No presenter-style gestures.
\u2022 No idle standing.
\u2022 No jitter.
\u2022 No flickering.
\u2022 No facial distortion.
\u2022 No identity drift.
\u2022 No hand deformation.
\u2022 No extra fingers.
\u2022 No malformed limbs.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. Camera movement follows the beat and amplifies the performance.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, every second feels alive, impossible to look away.

Spoken dialogue:
\"Open up the canvas, blank space on my screen. 
Drag a Checkpoint Loader, you know what I mean.
KSampler in the middle, VAE on the right,
Put the Text Encoder, yeah, building tonight.
Connect the nodes, run the queue,
Watch the latent flow right through.
Green, nothing green, nothing yellow,
Positive Prompt, in my hub.\" """

# -- Timeline Data (from JSON node 131 widgets_values[6]) --
# Contains 5 image segments and 1 audio segment
TIMELINE_DATA = json.dumps({
    "mainTrackEnabled": True,
    "audioTrackEnabled": True,
    "motionTrackEnabled": True,
    "propHeight": 90,
    "globalPropHeight": 470,
    "showFilenames": True,
    "overrideAudio": False,
    "inpaint_audio": True,
    "global_prompt": GLOBAL_PROMPT,
    "retake_global_prompt": "",
    "retakeMode": False,
    "retakeStart": 24,
    "retakeLength": 48,
    "retakePrompt": "",
    "retakeStrength": 1,
    "retakeVideo": None,
    "normalStartFrame": 0,
    "normalDurationFrames": 756,
    "segments": [
        {
            "id": "1785555235678s2fn3",
            "start": 0,
            "length": 226.01059340956584,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/1.png",
            "imageB64": "/api/view?filename=1.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "17855552413529uw9r",
            "start": 226.01059340956584,
            "length": 161.31859976617454,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/2.png",
            "imageB64": "/api/view?filename=2.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "1785555243885y3h85",
            "start": 387.3291931757404,
            "length": 131.45629831196658,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/3.png",
            "imageB64": "/api/view?filename=3.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "1785555247117rcoma",
            "start": 518.785491487707,
            "length": 225.5063328766255,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/4.png",
            "imageB64": "/api/view?filename=4.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        },
        {
            "id": "17855554543736wlrg",
            "start": 744.2918243643325,
            "length": 83.22765271847516,
            "prompt": "",
            "type": "image",
            "imageFile": "whatdreamscost/5.3.png",
            "imageB64": "/api/view?filename=5.3.png&type=input&subfolder=whatdreamscost",
            "isEndFrame": False
        }
    ],
    "motionSegments": [],
    "audioSegments": [
        {
            "id": "1785169457779kollx",
            "type": "audio",
            "start": 0,
            "length": 756.5194770828076,
            "trimStart": 446.9222739141953,
            "audioDurationFrames": 2880,
            "audioFile": "whatdreamscost/Late night trap.mp3",
            "fileName": "Late night trap.mp3",
            "waveformPeaks": []
        }
    ]
})

# -- Other LTXDirector parameters from JSON --
LOCAL_PROMPTS = " |  |  |  | "
SEGMENT_LENGTHS = "226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,11.708175635667544"
GUIDE_STRENGTH = "1.00,1.00,1.00,1.00,1.00"

# ============================================================
# Call LTXDirector node (node 131)
# Inputs:
#   model -> from ModelPreviewOverrideKJ (node 10)
#   clip -> from Power Lora Loader (node 138) CLIP output
#   audio_vae -> from VAELoader (node 8) audio VAE
#   optional_latent -> None (not connected in JSON)
#   global_prompt -> None input (uses internal property)
# Widgets_values order from JSON:
#   [0] start_second = 0
#   [1] end_second = 31.5
#   [2] duration_seconds = 31.5
#   [3] start_frame = 0
#   [4] end_frame = 756
#   [5] duration_frames = 756
#   [6] timeline_data (JSON string)
#   [7] local_prompts = " |  |  |  | "
#   [8] segment_lengths = "226.01...11.708..."
#   [9] epsilon = 0.001
#   [10] guide_strength = "1.00,1.00,1.00,1.00,1.00"
#   [11] mainTrackEnabled = True
#   [12] audioTrackEnabled = True
#   [13] motionTrackEnabled = True
#   [14] frame_rate = 24
#   [15] display_mode = "seconds"
#   [16] custom_width = 1280
#   [17] custom_height = 720
#   [18] resize_method = "maintain aspect ratio"
#   [19] divisible_by = 32
#   [20] img_compression = 18
#   [21] retakeMode = False
#   [22] "" (empty string)
# Outputs:
#   [0] model -> nodes 132, 133 (LTXDirectorGuide)
#   [1] positive -> nodes 27 (LTXVConditioning), 128 (ConditioningZeroOut)
#   [2] video_latent -> node 133 (LTXDirectorGuide Pass 1)
#   [3] audio_latent -> node 29 (LTXVConcatAVLatent Pass 1)
#   [4] guide_data -> nodes 132, 133
#   [5] motion_guide_data -> nodes 132, 133
#   [6] frame_rate -> nodes 27 (LTXVConditioning), 139 (VHS_VideoCombine)
#   [7] combined_audio -> not connected
# ============================================================
print("\nCalling LTXDirector (31.5s, 756 frames, 1280x720, 24fps)...")

ltxdirector = NODE_CLASS_MAPPINGS["LTXDirector"]()
ltxdirector_result = getattr(ltxdirector, ltxdirector.FUNCTION)(
    model=preview_model,
    clip=lora_clip,
    audio_vae=audio_vae,
    start_second=0,
    end_second=31.5,
    duration_seconds=31.5,
    start_frame=0,
    end_frame=756,
    duration_frames=756,
    timeline_data=TIMELINE_DATA,
    local_prompts=LOCAL_PROMPTS,
    segment_lengths=SEGMENT_LENGTHS,
    epsilon=0.001,
    guide_strength=GUIDE_STRENGTH,
    mainTrackEnabled=True,
    audioTrackEnabled=True,
    motionTrackEnabled=True,
    frame_rate=24,
    display_mode="seconds",
    custom_width=1280,
    custom_height=720,
    resize_method="maintain aspect ratio",
    divisible_by=32,
    img_compression=18,
    retakeMode=False,
    global_prompt=GLOBAL_PROMPT,
)

# Extract all 7 outputs from LTXDirector
director_model = get_value_at_index(ltxdirector_result, 0)       # MODEL
director_positive = get_value_at_index(ltxdirector_result, 1)    # positive CONDITIONING
director_video_latent = get_value_at_index(ltxdirector_result, 2)  # video LATENT
director_audio_latent = get_value_at_index(ltxdirector_result, 3)  # audio LATENT
director_guide_data = get_value_at_index(ltxdirector_result, 4)  # GUIDE_DATA
director_motion_guide_data = get_value_at_index(ltxdirector_result, 5)  # MOTION_GUIDE_DATA
director_frame_rate = get_value_at_index(ltxdirector_result, 6)  # FLOAT (24.0)

_vram_print("after LTXDirector")
print(f"   LTXDirector outputs ready. Frame rate: {director_frame_rate}")
print(f"   Duration: 31.5s, Frames: 756, Resolution: 1280x720")


# ======================================================================
# CELL 6 -- CONDITIONING & PASS 1 PIPELINE
# ======================================================================
# @title { "single-column": true }
# @markdown ## 6. Conditioning + Pass 1 (8 steps, euler, denoise=1.0)
# Pass 1: Low-res generation at DirectorGuide strength=0.5

print("=" * 60)
print("CELL 6: Conditioning & Pass 1 Pipeline")
print("=" * 60)

_vram_free()

# ============================================================
# Node 128: ConditioningZeroOut
# Input: positive conditioning from LTXDirector (node 131, output 1)
# Output: zeroed conditioning -> node 27 (negative input)
# Purpose: Creates a "null" negative conditioning for CFG
# ============================================================
print("\n[Pass1 1/9] ConditioningZeroOut on positive...")

conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
conditioningzeroout_result = conditioningzeroout.zero_out(
    conditioning=director_positive
)
zeroed_negative = get_value_at_index(conditioningzeroout_result, 0)
print("   Zeroed conditioning created (negative).")

# ============================================================
# Node 27: LTXVConditioning
# Inputs:
#   positive -> from LTXDirector (node 131, output 1)
#   negative -> from ConditioningZeroOut (node 128)
#   frame_rate -> from LTXDirector (node 131, output 6) = 24.0
# Widgets: frame_rate = 24
# Output:
#   positive -> node 133 (LTXDirectorGuide Pass 1)
#   negative -> node 133 (LTXDirectorGuide Pass 1)
# ============================================================
print("\n[Pass1 2/9] LTXVConditioning (wrapping pos/neg at 24fps)...")

ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
ltxvconditioning_result = ltxvconditioning.encode(
    positive=director_positive,
    negative=zeroed_negative,
    frame_rate=director_frame_rate
)
cond_positive = get_value_at_index(ltxvconditioning_result, 0)
cond_negative = get_value_at_index(ltxvconditioning_result, 1)
print("   LTXVConditioning applied (24fps frame rate encoding).")

# ============================================================
# Node 133: LTXDirectorGuide (Pass 1)
# Strength: 0.5
# Inputs:
#   positive -> from LTXVConditioning (node 27, output 0)
#   negative -> from LTXVConditioning (node 27, output 1)
#   vae -> video_vae (node 36)
#   latent -> video_latent from LTXDirector (node 131, output 2)
#   guide_data -> from LTXDirector (node 131, output 4)
#   motion_guide_data -> from LTXDirector (node 131, output 5)
#   model -> from LTXDirector (node 131, output 0)
# Widgets: ["None", 1, 0.5, "bicubic", 1, "center", true, false, 256, 64, false]
#   [0] preset = "None"
#   [1] scale_factor = 1
#   [2] strength = 0.5
#   [3] interpolation = "bicubic"
#   [4] temporal_scale = 1
#   [5] alignment = "center"
#   [6] pad_to_mult = True
#   [7] crop_output = False
#   [8] tile_size = 256
#   [9] tile_overlap = 64
#   [10] use_tiling = False
# Outputs:
#   [0] positive -> node 28 (CFGGuider P1), node 55 (CropGuides between passes)
#   [1] negative -> node 28 (CFGGuider P1), node 55 (CropGuides between passes)
#   [2] latent -> node 29 (LTXVConcatAVLatent P1)
#   [3] model -> node 33 (BasicScheduler P1), node 28 (CFGGuider P1)
# ============================================================
print("\n[Pass1 3/9] LTXDirectorGuide Pass 1 (strength=0.5)...")

ltxdirectorguide_p1 = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
ltxdirectorguide_p1_result = getattr(ltxdirectorguide_p1, ltxdirectorguide_p1.FUNCTION)(
    positive=cond_positive,
    negative=cond_negative,
    vae=video_vae,
    latent=director_video_latent,
    guide_data=director_guide_data,
    motion_guide_data=director_motion_guide_data,
    model=director_model,
    preset="None",
    scale_factor=1,
    strength=0.5,
    interpolation="bicubic",
    temporal_scale=1,
    alignment="center",
    pad_to_mult=True,
    crop_output=False,
    tile_size=256,
    tile_overlap=64,
    use_tiling=False,
)
p1_positive = get_value_at_index(ltxdirectorguide_p1_result, 0)
p1_negative = get_value_at_index(ltxdirectorguide_p1_result, 1)
p1_video_latent = get_value_at_index(ltxdirectorguide_p1_result, 2)
p1_model = get_value_at_index(ltxdirectorguide_p1_result, 3)

_vram_print("after DirectorGuide P1")
print("   DirectorGuide Pass 1 configured (strength=0.5).")

# ============================================================
# Node 29: LTXVConcatAVLatent (Pass 1)
# Combines video_latent (from DirectorGuide 133) + audio_latent (from LTXDirector 131)
# Input:
#   video_latent -> from LTXDirectorGuide P1 (node 133, output 2)
#   audio_latent -> from LTXDirector (node 131, output 3)
# Output: combined latent -> node 31 (SamplerCustomAdvanced P1, latent_image)
# ============================================================
print("\n[Pass1 4/9] LTXVConcatAVLatent Pass 1...")

ltxvconcatavlatent_p1 = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
ltxvconcatavlatent_p1_result = ltxvconcatavlatent_p1.concat(
    video_latent=p1_video_latent,
    audio_latent=director_audio_latent
)
p1_combined_latent = get_value_at_index(ltxvconcatavlatent_p1_result, 0)
print("   Video + Audio latents concatenated for Pass 1.")

# ============================================================
# Node 28: CFGGuider (Pass 1)
# cfg = 1.0 (from JSON widgets_values: [1])
# Inputs:
#   model -> from LTXDirectorGuide P1 (node 133, output 3)
#   positive -> from LTXDirectorGuide P1 (node 133, output 0)
#   negative -> from LTXDirectorGuide P1 (node 133, output 1)
# Output: GUIDER -> node 31 (SamplerCustomAdvanced P1)
# ============================================================
print("\n[Pass1 5/9] CFGGuider Pass 1 (cfg=1.0)...")

cfgguider_p1 = NODE_CLASS_MAPPINGS["CFGGuider"]()
cfgguider_p1_result = cfgguider_p1.get_guider(
    model=p1_model,
    positive=p1_positive,
    negative=p1_negative,
    cfg=1.0
)
p1_guider = get_value_at_index(cfgguider_p1_result, 0)
print("   CFGGuider Pass 1 ready (cfg=1.0).")

# ============================================================
# Node 32: KSamplerSelect (Pass 1)
# sampler_name = "euler"
# Output: SAMPLER -> node 31 (SamplerCustomAdvanced P1)
# ============================================================
print("\n[Pass1 6/9] KSamplerSelect Pass 1 (euler)...")

ksamplerselect_p1 = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
ksamplerselect_p1_result = ksamplerselect_p1.get_sampler(
    sampler_name="euler"
)
p1_sampler = get_value_at_index(ksamplerselect_p1_result, 0)
print("   Euler sampler selected for Pass 1.")

# ============================================================
# Node 33: BasicScheduler (Pass 1)
# scheduler = "linear_quadratic", steps = 8, denoise = 1.0
# Input: model -> from LTXDirectorGuide P1 (node 133, output 3)
# Output: SIGMAS -> node 31 (SamplerCustomAdvanced P1)
# ============================================================
print("\n[Pass1 7/9] BasicScheduler Pass 1 (linear_quadratic, 8 steps, denoise=1.0)...")

basicscheduler_p1 = NODE_CLASS_MAPPINGS["BasicScheduler"]()
basicscheduler_p1_result = basicscheduler_p1.get_sigmas(
    model=p1_model,
    scheduler="linear_quadratic",
    steps=8,
    denoise=1.0
)
p1_sigmas = get_value_at_index(basicscheduler_p1_result, 0)
print("   Sigmas computed: 8 steps, linear_quadratic, full denoise.")

# ============================================================
# Node 30: RandomNoise
# seed = 0, control_after_generate = "fixed"
# Output: NOISE -> node 31 (SamplerCustomAdvanced P1)
#                -> node 19 (SamplerCustomAdvanced P2) [shared noise]
# ============================================================
print("\n[Pass1 8/9] RandomNoise (seed=0, fixed)...")

randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
randomnoise_result = randomnoise.get_noise(
    noise_seed=0
)
noise = get_value_at_index(randomnoise_result, 0)
print("   Random noise generated (seed=0, shared between passes).")

# ============================================================
# Node 31: SamplerCustomAdvanced (Pass 1)
# Inputs:
#   noise -> from RandomNoise (node 30)
#   guider -> from CFGGuider P1 (node 28)
#   sampler -> from KSamplerSelect P1 (node 32)
#   sigmas -> from BasicScheduler P1 (node 33)
#   latent_image -> from LTXVConcatAVLatent P1 (node 29)
# Outputs:
#   [0] output -> node 34 (LTXVSeparateAVLatent, between passes)
#   [1] denoised_output -> not connected
# ============================================================
print("\n[Pass1 9/9] SamplerCustomAdvanced Pass 1 (8 steps, full denoise)...")
_vram_print("before sampling P1")

samplercustomadvanced_p1 = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
samplercustomadvanced_p1_result = samplercustomadvanced_p1.sample(
    noise=noise,
    guider=p1_guider,
    sampler=p1_sampler,
    sigmas=p1_sigmas,
    latent_image=p1_combined_latent
)
p1_output = get_value_at_index(samplercustomadvanced_p1_result, 0)

_vram_free()
_vram_print("after Pass 1 sampling")
print("   Pass 1 sampling complete.")


# ======================================================================
# CELL 7 -- PASS 2 PIPELINE (UPSCALE + REFINE)
# ======================================================================
# @title { "single-column": true }
# @markdown ## 7. Pass 2 (4 steps, euler, denoise=0.42, upscaled)
# Pass 2: High-res refinement at DirectorGuide strength=1.0

print("=" * 60)
print("CELL 7: Pass 2 Pipeline (Upscale + Refine)")
print("=" * 60)

_vram_free()

# ============================================================
# Node 34: LTXVSeparateAVLatent (between passes)
# Splits Pass 1 output into video + audio latents
# Input: av_latent -> from SamplerCustomAdvanced P1 (node 31, output 0)
# Outputs:
#   [0] video_latent -> node 55 (LTXDirectorCropGuides, latent input)
#   [1] audio_latent -> node 18 (LTXVConcatAVLatent P2)
# ============================================================
print("\n[Pass2 1/9] LTXVSeparateAVLatent (split Pass 1 output)...")

ltxvseparateavlatent_between = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
ltxvseparateavlatent_between_result = ltxvseparateavlatent_between.separate(
    av_latent=p1_output
)
between_video_latent = get_value_at_index(ltxvseparateavlatent_between_result, 0)
between_audio_latent = get_value_at_index(ltxvseparateavlatent_between_result, 1)
print("   Pass 1 output separated into video + audio latents.")

# ============================================================
# Node 55: LTXDirectorCropGuides (between passes)
# Crops/adjusts conditioning to match the latent spatial dimensions
# Inputs:
#   positive -> from LTXDirectorGuide P1 (node 133, output 0)
#   negative -> from LTXDirectorGuide P1 (node 133, output 1)
#   latent -> video_latent from LTXVSeparateAVLatent (node 34, output 0)
# Outputs:
#   [0] positive -> node 132 (LTXDirectorGuide P2)
#   [1] negative -> node 132 (LTXDirectorGuide P2)
#   [2] latent -> node 14 (LTXVLatentUpsampler)
# ============================================================
print("\n[Pass2 2/9] LTXDirectorCropGuides (between passes, node 55)...")

ltxdirectorcropguides_between = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
ltxdirectorcropguides_between_result = getattr(
    ltxdirectorcropguides_between, ltxdirectorcropguides_between.FUNCTION
)(
    positive=p1_positive,
    negative=p1_negative,
    latent=between_video_latent
)
cropped_positive = get_value_at_index(ltxdirectorcropguides_between_result, 0)
cropped_negative = get_value_at_index(ltxdirectorcropguides_between_result, 1)
cropped_latent = get_value_at_index(ltxdirectorcropguides_between_result, 2)
print("   Guides cropped for upscale pass.")

# ============================================================
# Node 14: LTXVLatentUpsampler
# 2x spatial upscale of the latent
# Inputs:
#   samples -> cropped latent from LTXDirectorCropGuides (node 55, output 2)
#   upscale_model -> from LatentUpscaleModelLoader (node 13)
#   vae -> video_vae (node 36)
# Output: LATENT -> node 132 (LTXDirectorGuide P2, latent input)
# ============================================================
print("\n[Pass2 3/9] LTXVLatentUpsampler (2x spatial upscale)...")
_vram_print("before upscale")

ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
ltxvlatentupsampler_result = ltxvlatentupsampler.upsample(
    samples=cropped_latent,
    upscale_model=upscale_model,
    vae=video_vae
)
upscaled_latent = get_value_at_index(ltxvlatentupsampler_result, 0)

_vram_print("after upscale")
print("   Latent upscaled 2x.")

# ============================================================
# Node 132: LTXDirectorGuide (Pass 2)
# Strength: 1.0
# Inputs:
#   positive -> from LTXDirectorCropGuides (node 55, output 0)
#   negative -> from LTXDirectorCropGuides (node 55, output 1)
#   vae -> video_vae (node 36)
#   latent -> upscaled latent from LTXVLatentUpsampler (node 14)
#   guide_data -> from LTXDirector (node 131, output 4)
#   motion_guide_data -> from LTXDirector (node 131, output 5)
#   model -> from LTXDirector (node 131, output 0)
# Widgets: ["None", 1, 1, "bicubic", 1, "center", true, false, 256, 64, false]
#   [0] preset = "None"
#   [1] scale_factor = 1
#   [2] strength = 1.0
#   [3] interpolation = "bicubic"
#   [4] temporal_scale = 1
#   [5] alignment = "center"
#   [6] pad_to_mult = True
#   [7] crop_output = False
#   [8] tile_size = 256
#   [9] tile_overlap = 64
#   [10] use_tiling = False
# Outputs:
#   [0] positive -> node 17 (CFGGuider P2), node 54 (CropGuides after P2)
#   [1] negative -> node 17 (CFGGuider P2), node 54 (CropGuides after P2)
#   [2] latent -> node 18 (LTXVConcatAVLatent P2)
#   [3] model -> node 21 (BasicScheduler P2), node 17 (CFGGuider P2)
# ============================================================
print("\n[Pass2 4/9] LTXDirectorGuide Pass 2 (strength=1.0)...")

ltxdirectorguide_p2 = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
ltxdirectorguide_p2_result = getattr(ltxdirectorguide_p2, ltxdirectorguide_p2.FUNCTION)(
    positive=cropped_positive,
    negative=cropped_negative,
    vae=video_vae,
    latent=upscaled_latent,
    guide_data=director_guide_data,
    motion_guide_data=director_motion_guide_data,
    model=director_model,
    preset="None",
    scale_factor=1,
    strength=1.0,
    interpolation="bicubic",
    temporal_scale=1,
    alignment="center",
    pad_to_mult=True,
    crop_output=False,
    tile_size=256,
    tile_overlap=64,
    use_tiling=False,
)
p2_positive = get_value_at_index(ltxdirectorguide_p2_result, 0)
p2_negative = get_value_at_index(ltxdirectorguide_p2_result, 1)
p2_video_latent = get_value_at_index(ltxdirectorguide_p2_result, 2)
p2_model = get_value_at_index(ltxdirectorguide_p2_result, 3)

_vram_print("after DirectorGuide P2")
print("   DirectorGuide Pass 2 configured (strength=1.0).")

# ============================================================
# Node 18: LTXVConcatAVLatent (Pass 2)
# Combines video_latent (from DirectorGuide 132) + audio_latent (from Separate 34)
# Input:
#   video_latent -> from LTXDirectorGuide P2 (node 132, output 2)
#   audio_latent -> from LTXVSeparateAVLatent (node 34, output 1)
# Output: combined latent -> node 19 (SamplerCustomAdvanced P2, latent_image)
# ============================================================
print("\n[Pass2 5/9] LTXVConcatAVLatent Pass 2...")

ltxvconcatavlatent_p2 = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
ltxvconcatavlatent_p2_result = ltxvconcatavlatent_p2.concat(
    video_latent=p2_video_latent,
    audio_latent=between_audio_latent
)
p2_combined_latent = get_value_at_index(ltxvconcatavlatent_p2_result, 0)
print("   Video + Audio latents concatenated for Pass 2.")

# ============================================================
# Node 17: CFGGuider (Pass 2)
# cfg = 1.0 (from JSON widgets_values: [1])
# Inputs:
#   model -> from LTXDirectorGuide P2 (node 132, output 3)
#   positive -> from LTXDirectorGuide P2 (node 132, output 0)
#   negative -> from LTXDirectorGuide P2 (node 132, output 1)
# Output: GUIDER -> node 19 (SamplerCustomAdvanced P2)
# ============================================================
print("\n[Pass2 6/9] CFGGuider Pass 2 (cfg=1.0)...")

cfgguider_p2 = NODE_CLASS_MAPPINGS["CFGGuider"]()
cfgguider_p2_result = cfgguider_p2.get_guider(
    model=p2_model,
    positive=p2_positive,
    negative=p2_negative,
    cfg=1.0
)
p2_guider = get_value_at_index(cfgguider_p2_result, 0)
print("   CFGGuider Pass 2 ready (cfg=1.0).")

# ============================================================
# Node 20: KSamplerSelect (Pass 2)
# sampler_name = "euler"
# Output: SAMPLER -> node 19 (SamplerCustomAdvanced P2)
# ============================================================
print("\n[Pass2 7/9] KSamplerSelect Pass 2 (euler)...")

ksamplerselect_p2 = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
ksamplerselect_p2_result = ksamplerselect_p2.get_sampler(
    sampler_name="euler"
)
p2_sampler = get_value_at_index(ksamplerselect_p2_result, 0)
print("   Euler sampler selected for Pass 2.")

# ============================================================
# Node 21: BasicScheduler (Pass 2)
# scheduler = "linear_quadratic", steps = 4, denoise = 0.42
# Input: model -> from LTXDirectorGuide P2 (node 132, output 3)
# Output: SIGMAS -> node 19 (SamplerCustomAdvanced P2)
# ============================================================
print("\n[Pass2 8/9] BasicScheduler Pass 2 (linear_quadratic, 4 steps, denoise=0.42)...")

basicscheduler_p2 = NODE_CLASS_MAPPINGS["BasicScheduler"]()
basicscheduler_p2_result = basicscheduler_p2.get_sigmas(
    model=p2_model,
    scheduler="linear_quadratic",
    steps=4,
    denoise=0.42
)
p2_sigmas = get_value_at_index(basicscheduler_p2_result, 0)
print("   Sigmas computed: 4 steps, linear_quadratic, denoise=0.42.")

# ============================================================
# Node 19: SamplerCustomAdvanced (Pass 2)
# Uses SAME noise as Pass 1 (node 30 connects to both node 31 and node 19)
# Inputs:
#   noise -> from RandomNoise (node 30) [same as Pass 1]
#   guider -> from CFGGuider P2 (node 17)
#   sampler -> from KSamplerSelect P2 (node 20)
#   sigmas -> from BasicScheduler P2 (node 21)
#   latent_image -> from LTXVConcatAVLatent P2 (node 18)
# Outputs:
#   [0] output -> node 22 (LTXVSeparateAVLatent, after Pass 2)
#   [1] denoised_output -> not connected
# ============================================================
print("\n[Pass2 9/9] SamplerCustomAdvanced Pass 2 (4 steps, denoise=0.42)...")
_vram_print("before sampling P2")

samplercustomadvanced_p2 = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
samplercustomadvanced_p2_result = samplercustomadvanced_p2.sample(
    noise=noise,
    guider=p2_guider,
    sampler=p2_sampler,
    sigmas=p2_sigmas,
    latent_image=p2_combined_latent
)
p2_output = get_value_at_index(samplercustomadvanced_p2_result, 0)

_vram_free()
_vram_print("after Pass 2 sampling")
print("   Pass 2 sampling complete (upscaled + refined).")


# ======================================================================
# CELL 8 -- DECODE & OUTPUT
# ======================================================================
# @title { "single-column": true }
# @markdown ## 8. Decode Video + Audio, Combine Output
# Final output: h264-mp4, CRF=8, yuv420p, 24fps

print("=" * 60)
print("CELL 8: Decode & Output")
print("=" * 60)

_vram_free()

# ============================================================
# Node 22: LTXVSeparateAVLatent (after Pass 2)
# Splits Pass 2 output into video + audio latents
# Input: av_latent -> from SamplerCustomAdvanced P2 (node 19, output 0)
# Outputs:
#   [0] video_latent -> node 54 (LTXDirectorCropGuides after P2)
#   [1] audio_latent -> node 24 (LTXVAudioVAEDecode)
# ============================================================
print("\n[Output 1/5] LTXVSeparateAVLatent (split Pass 2 output)...")

ltxvseparateavlatent_final = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
ltxvseparateavlatent_final_result = ltxvseparateavlatent_final.separate(
    av_latent=p2_output
)
final_video_latent = get_value_at_index(ltxvseparateavlatent_final_result, 0)
final_audio_latent = get_value_at_index(ltxvseparateavlatent_final_result, 1)
print("   Pass 2 output separated into video + audio latents.")

# ============================================================
# Node 54: LTXDirectorCropGuides (after Pass 2)
# Crops conditioning to match final video latent dimensions
# Inputs:
#   positive -> from LTXDirectorGuide P2 (node 132, output 0)
#   negative -> from LTXDirectorGuide P2 (node 132, output 1)
#   latent -> video_latent from LTXVSeparateAVLatent (node 22, output 0)
# Outputs:
#   [0] positive -> not connected (null links in JSON)
#   [1] negative -> not connected (null links in JSON)
#   [2] latent -> node 1 (VAEDecode)
# ============================================================
print("\n[Output 2/5] LTXDirectorCropGuides (after Pass 2, node 54)...")

ltxdirectorcropguides_final = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
ltxdirectorcropguides_final_result = getattr(
    ltxdirectorcropguides_final, ltxdirectorcropguides_final.FUNCTION
)(
    positive=p2_positive,
    negative=p2_negative,
    latent=final_video_latent
)
final_cropped_latent = get_value_at_index(ltxdirectorcropguides_final_result, 2)
print("   Final guides cropped for VAE decode.")

# ============================================================
# Node 1: VAEDecode
# Decodes video latent to pixel space
# Inputs:
#   samples -> cropped latent from LTXDirectorCropGuides (node 54, output 2)
#   vae -> video_vae (node 36)
# Output: IMAGE -> node 139 (VHS_VideoCombine)
# ============================================================
print("\n[Output 3/5] VAEDecode (video latent -> pixels)...")
_vram_print("before VAE decode")

vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
vaedecode_result = vaedecode.decode(
    samples=final_cropped_latent,
    vae=video_vae
)
video_images = get_value_at_index(vaedecode_result, 0)

_vram_print("after VAE decode")
print("   Video decoded to pixel space.")

# ============================================================
# Node 24: LTXVAudioVAEDecode
# Decodes audio latent to audio waveform
# Inputs:
#   samples -> audio_latent from LTXVSeparateAVLatent (node 22, output 1)
#   audio_vae -> from VAELoader (node 8)
# Output: AUDIO -> node 139 (VHS_VideoCombine)
# ============================================================
print("\n[Output 4/5] LTXVAudioVAEDecode (audio latent -> waveform)...")

ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
ltxvaudiovaedecode_result = ltxvaudiovaedecode.decode(
    samples=final_audio_latent,
    audio_vae=audio_vae
)
audio_output = get_value_at_index(ltxvaudiovaedecode_result, 0)
print("   Audio decoded.")

_vram_free()

# ============================================================
# Node 139: VHS_VideoCombine
# Combines decoded video frames + audio into final mp4
# Inputs:
#   images -> from VAEDecode (node 1)
#   audio -> from LTXVAudioVAEDecode (node 24)
#   frame_rate -> from LTXDirector (node 131, output 6) = 24.0
# Widgets (from JSON):
#   frame_rate = 24
#   loop_count = 0
#   filename_prefix = "LTX2.3/Video"
#   format = "video/h264-mp4"
#   pix_fmt = "yuv420p"
#   crf = 8
#   save_metadata = False
#   trim_to_audio = False
#   pingpong = False
#   save_output = True
# Output: VHS_FILENAMES (final video path)
# ============================================================
print("\n[Output 5/5] VHS_VideoCombine (h264-mp4, CRF=8, 24fps)...")

vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
vhs_videocombine_result = vhs_videocombine.combine_video(
    images=video_images,
    audio=audio_output,
    frame_rate=director_frame_rate,
    loop_count=0,
    filename_prefix="LTX2.3/Video",
    format="video/h264-mp4",
    pix_fmt="yuv420p",
    crf=8,
    save_metadata=False,
    trim_to_audio=False,
    pingpong=False,
    save_output=True,
)
output_filenames = get_value_at_index(vhs_videocombine_result, 0)

_vram_free()
_vram_print("after output")
print(f"\n   Video saved: {output_filenames}")
print("   Format: h264-mp4, CRF=8, yuv420p, 24fps")
print("   Output complete!")


# ======================================================================
# CELL 9 -- CONFIGURATION & RUN
# ======================================================================
# @title { "single-column": true }
# @markdown ## 9. Configuration & Run
# User-editable parameters for the music video pipeline.
# Modify these before running the pipeline.

# ============================================================
# USER CONFIGURATION
# ============================================================

CONFIG = {
    # -- Generation parameters --
    "seed": 0,                          # Random seed (0 = random)
    "duration_seconds": 31.5,           # Total video duration
    "frame_rate": 24,                   # Frames per second
    "custom_width": 1280,               # Video width
    "custom_height": 720,               # Video height
    "divisible_by": 32,                 # Resolution must be divisible by this

    # -- Pipeline parameters --
    "pass1_steps": 8,                   # Pass 1 sampling steps
    "pass1_denoise": 1.0,              # Pass 1 denoise strength (full)
    "pass1_cfg": 1.0,                  # Pass 1 CFG scale
    "pass1_guide_strength": 0.5,       # DirectorGuide Pass 1 strength
    "pass2_steps": 4,                   # Pass 2 sampling steps
    "pass2_denoise": 0.42,             # Pass 2 denoise strength (partial)
    "pass2_cfg": 1.0,                  # Pass 2 CFG scale
    "pass2_guide_strength": 1.0,       # DirectorGuide Pass 2 strength
    "sampler": "euler",                 # Sampler name
    "scheduler": "linear_quadratic",    # Scheduler type

    # -- Image segments (reference images for the timeline) --
    "image_segments": [
        {"file": "whatdreamscost/1.png", "start": 0, "length": 226.01059340956584},
        {"file": "whatdreamscost/2.png", "start": 226.01059340956584, "length": 161.31859976617454},
        {"file": "whatdreamscost/3.png", "start": 387.3291931757404, "length": 131.45629831196658},
        {"file": "whatdreamscost/4.png", "start": 518.785491487707, "length": 225.5063328766255},
        {"file": "whatdreamscost/5.3.png", "start": 744.2918243643325, "length": 83.22765271847516},
    ],

    # -- Audio segment --
    "audio_file": "whatdreamscost/Late night trap.mp3",
    "audio_trim_start": 446.9222739141953,
    "audio_duration_frames": 2880,

    # -- LoRA configuration --
    "loras": [
        {"name": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", "strength": 0.4},
        {"name": "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", "strength": 0.6},
        {"name": "ltx2.3-transition.safetensors", "strength": 0.7},
        {"name": "LTX2.3-MVCamera-drclips.safetensors", "strength": 0.9},
    ],

    # -- Director parameters --
    "guide_strength": "1.00,1.00,1.00,1.00,1.00",
    "local_prompts": " |  |  |  | ",
    "segment_lengths": "226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,11.708175635667544",
    "resize_method": "maintain aspect ratio",
    "img_compression": 18,
    "epsilon": 0.001,

    # -- Output settings --
    "output_format": "video/h264-mp4",
    "output_pix_fmt": "yuv420p",
    "output_crf": 8,
    "output_prefix": "LTX2.3/Video",
    "save_output": True,
    "trim_to_audio": False,
}


def _run():
    """
    Execute the full LTX 2.3 Director Music Video pipeline.
    
    This function runs all cells in sequence:
    1. Environment setup (Cell 1) - should already be done
    2. Model downloads (Cell 2) - should already be done
    3. Core imports (Cell 3) - should already be done
    4. Model loading (Cell 4)
    5. LTXDirector setup (Cell 5)
    6. Conditioning & Pass 1 (Cell 6)
    7. Pass 2 Upscale + Refine (Cell 7)
    8. Decode & Output (Cell 8)
    
    The pipeline produces a 31.5-second music video at 1280x720, 24fps
    with synchronized audio from the timeline configuration.
    """
    print("=" * 60)
    print("LTX 2.3 Director 2.0 - Music Video Pipeline")
    print("=" * 60)
    print(f"  Resolution: {CONFIG['custom_width']}x{CONFIG['custom_height']}")
    print(f"  Duration: {CONFIG['duration_seconds']}s @ {CONFIG['frame_rate']}fps")
    print(f"  Pass 1: {CONFIG['pass1_steps']} steps, denoise={CONFIG['pass1_denoise']}")
    print(f"  Pass 2: {CONFIG['pass2_steps']} steps, denoise={CONFIG['pass2_denoise']}")
    print(f"  LoRAs: {len(CONFIG['loras'])} active")
    print(f"  Output: {CONFIG['output_format']}, CRF={CONFIG['output_crf']}")
    print("=" * 60)
    print("\nPipeline execution follows the cell-by-cell structure above.")
    print("Run each cell sequentially in Google Colab for full control,")
    print("or call this function after all cells have been executed.")
    print("\nTo run in Colab: Execute cells 1-8 in order.")
    print("Each cell is independent and can be re-run individually.")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    _run()
