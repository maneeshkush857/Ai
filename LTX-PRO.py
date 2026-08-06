# ======================================================================
# LTX-2 LD -- INFINITE FLOW ENGINE PRO (Unified Edition)
#
# Architecture:
#   CELL 1   Environment Setup (subprocess-based, no Colab magic)
#   CELL 2   Model Downloads (aria2c)
#   CELL 3   Core Imports & Utilities
#   SECTION A  VisionDescribeEngine (standalone Qwen2.5-VL)
#   SECTION B  EasyPromptEngine (cinematic LLM expansion + story workflow)
#   SECTION C  CharacterBible (cross-scene consistency lock)
#   SECTION D  generate_clip() (two-pass LTX-2 19B GGUF pipeline)
#   SECTION E  InfiniteFlowEngine (scene_chain + temporal_extend + svi_pro_extend)
#   VoiceSyncHook (placeholder for future audio integration)
#   CELL 4   Configuration
#   CELL 5   Run
#
# GPU target: Google Colab T4 (15 GB) / L4 (24 GB)
# Model: LTX-2 19B Q4_K_M GGUF (Kijai distilled)
# Strategy: load -> use -> unload every heavyweight model in sequence
#
# Key enhancements over previous version:
#   - SVI-Pro temporal extension with OVERLAP_FRAMES (16 frames)
#   - I2V re-conditioning in Pass 2 (LD-I2V.json pattern)
#   - Adaptive anchor strength for character consistency
#   - Story-to-prompt workflow (analyze, extract characters, generate scenes)
#   - Camera direction mapping with LoRA integration
#   - VoiceSyncHook foundation for future audio sync
#   - Proper VRAM management with explicit del + gc after each model
# ======================================================================

# ======================================================================
# CELL 1 -- ENVIRONMENT SETUP (run once per Colab session)
# ======================================================================
# @title { "single-column": true }
# @markdown ## 1. Install Environment

import subprocess
import os
import sys

# CELL 1  --  ENVIRONMENT SETUP  (run once per Colab session)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 1. Install Environment

import subprocess, os, sys

def _shell(cmd):
    """Run a shell command (replaces ! magic for AST compatibility)."""
    subprocess.run(cmd, shell=True, check=True)

def _pip(packages):
    """pip install helper."""
    _shell(f"pip install -q {packages}")

_shell("pip install torch torchvision torchaudio")


# ── Base Python packages ──────────────────────────────────────────────────────
!pip install torch torchvision torchaudio

%cd /content
from IPython.display import clear_output
clear_output()

!pip install -q torchsde einops diffusers accelerate nest_asyncio
!pip install -q av spandrel albumentations onnx opencv-python onnxruntime
!pip install -q imageio imageio-ffmpeg

# Extra packages required by EasyPrompt & VisionDescribe nodes
!pip install -q transformers>=4.43.0 accelerate qwen-vl-utils huggingface_hub

# ── ComfyUI (pinned branch — matches reference notebook) ─────────────────────
!git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt -q
clear_output()

# ── Custom nodes ──────────────────────────────────────────────────────────────
%cd /content/ComfyUI/custom_nodes

# Core KJNodes (pinned build — ImageResizeKJv2, PathchSageAttentionKJ, etc.)
!git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
# GGUF loader (UnetLoaderGGUF)
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git
# LTXVideo nodes (LTXVImgToVideoInplace, LTXVPreprocess, tiled VAE, etc.)
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
# LTX2EasyPrompt-LD — LTX2PromptArchitect + LTX2VisionDescribe
!git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
# LTX2-Master-Loader — LTX2MasterLoaderLD (10-slot LoRA stacker)
!git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git
# VideoHelperSuite — VHS_VideoCombine (h264-mp4, crf=19, yuv420p)
!git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# ── Install node requirements ─────────────────────────────────────────────────
%cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI-LTXVideo
!pip install -r requirements.txt -q 2>/dev/null || true

subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

os.chdir("/content/ComfyUI")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")
clear_output()
print("Environment ready. All custom nodes installed.")




%cd /content/ComfyUI/custom_nodes

import subprocess
import os

# Base nodes
!git clone https://github.com/kijai/ComfyUI-KJNodes
!git clone https://github.com/city96/ComfyUI-GGUF
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo/

# Custom Node Packs (LFS skip for fast cloning)
!GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/vidfom/Ltx-3 /content/Ltx-3-repo
!mv /content/Ltx-3-repo/ComfyUI/custom_nodes/LTX2EasyPrompt-LD /content/ComfyUI/custom_nodes/
!mv /content/Ltx-3-repo/ComfyUI/custom_nodes/LTX2-Master-Loader /content/ComfyUI/custom_nodes/
!rm -rf /content/Ltx-3-repo

# Install custom node requirements
%cd /content/ComfyUI/custom_nodes/ComfyUI-KJNodes
!pip install -r requirements.txt

%cd /content/ComfyUI/custom_nodes/ComfyUI-GGUF
!pip install -r requirements.txt

%cd /content/ComfyUI/custom_nodes/ComfyUI-LTXVideo
if os.path.exists("requirements.txt"):
    !pip install -r requirements.txt

%cd /content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD
if os.path.exists("requirements.txt"):
    !pip install -r requirements.txt

%cd /content/ComfyUI/custom_nodes/LTX2-Master-Loader
if os.path.exists("requirements.txt"):
    !pip install -r requirements.txt

def install_apt_packages():
    packages = ['aria2', 'ffmpeg']
    try:
        subprocess.run(['apt-get', '-y', 'install', '-qq'] + packages, check=True, capture_output=True)
        print("✓ apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing apt packages: {e.stderr.decode().strip() or 'Unknown error'}")

print("Installing apt packages...")
install_apt_packages()
clear_output()

%cd /content/ComfyUI



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
    """Download a file with aria2c, skip if cached and >1MB."""
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
        raise RuntimeError(
            f"Download failed for {fn}: {r.stderr.strip() or 'unknown error'}. "
            f"Check network connection and re-run Cell 2.")
    print("done.")
    return fn


KIJAI = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
CORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LTX_HF = "https://huggingface.co/Lightricks"
UNET_D = "/content/ComfyUI/models/unet"
TE_D = "/content/ComfyUI/models/text_encoders"
VAE_D = "/content/ComfyUI/models/vae"
UPD = "/content/ComfyUI/models/latent_upscale_models"
LR_D = "/content/ComfyUI/models/loras"

print("-- Core models --")
_M_UNET = _dl(f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf", UNET_D)
_M_CLIP1 = _dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", TE_D)
_M_CLIP2 = _dl(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", TE_D
)
_M_VAE = _dl(f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors", VAE_D)
_M_AVAE = _dl(f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors", VAE_D)
_M_TAEV = _dl(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    VAE_D,
)
_M_UP = _dl(
    f"{LTX_HF}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors", UPD
)

print("\n-- LoRAs --")
_LORA_URLS = {
    "Detailer": f"{LTX_HF}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny": f"{LTX_HF}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth": f"{LTX_HF}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose": f"{LTX_HF}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Out": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Left": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Right": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
}
os.makedirs(LR_D, exist_ok=True)
for name, url in _LORA_URLS.items():
    _dl(url, LR_D)

print("\nAll models ready.")



# ======================================================================
# CELL 3 -- CORE IMPORTS & UTILITIES
# ======================================================================
# @title { "single-column": true }
# @markdown ## 3. Imports, Helpers, Node Loader

import gc
import re
import json
import time
import shutil
import warnings
import asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping, Dict, Tuple
from base64 import b64encode
from IPython.display import display, HTML, clear_output
try:
    from google.colab import files
except ImportError:
    files = None

warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/ComfyUI")
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# -- PRO v2 Constants --
OVERLAP_FRAMES = 16
ANCHOR_STRENGTH_HIGH = 0.85
ANCHOR_STRENGTH_LOW = 0.70
USE_ADAPTIVE_STRENGTH = True


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
# Tensor / Image Helpers
# ======================================================================

def get_value_at_index(obj: Union[Sequence, Mapping], idx: int) -> Any:
    """Extract a value from ComfyUI node output (list or dict)."""
    try:
        return obj[idx]
    except KeyError:
        return obj["result"][idx]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float32 tensor [1, H, W, 3] in 0..1 range."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor [1, H, W, 3] or [H, W, 3] to PIL Image."""
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    """Load an image file as a float32 tensor, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))


def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    """Extract the last frame of a video as a float32 tensor."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


def extract_overlap_frames(video_path: str, n_frames: int = OVERLAP_FRAMES) -> Optional[torch.Tensor]:
    """
    Extract the last N frames from a video as a batch tensor [N, H, W, 3].
    Used by SVI-Pro extension mode for multi-frame conditioning.
    Returns None if video cannot be read or has fewer frames than requested.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return None
    start_frame = total - n_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) < n_frames:
        return None
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def blend_anchor_with_prev(
    anchor: torch.Tensor,
    prev_frames: torch.Tensor,
    weight: float = 0.25,
) -> torch.Tensor:
    """
    Blend anchor image with previous frames in pixel space.
    anchor: [1, H, W, 3] - the original seed/reference image
    prev_frames: [N, H, W, 3] or [1, H, W, 3] - frames from previous clip
    weight: blend factor (0.0 = pure prev_frames, 1.0 = pure anchor)
    Returns: blended tensor matching prev_frames shape (uses last frame only for single output).
    """
    if prev_frames.ndim == 4 and prev_frames.shape[0] > 1:
        # Use only the last frame for blending with anchor
        prev_single = prev_frames[-1:, :, :, :]
    else:
        prev_single = prev_frames
    # Ensure anchor and prev have compatible spatial dims
    if anchor.shape[1:3] != prev_single.shape[1:3]:
        # Resize anchor to match prev_single dimensions via interpolation
        a = anchor.permute(0, 3, 1, 2)
        a = torch.nn.functional.interpolate(
            a, size=(prev_single.shape[1], prev_single.shape[2]), mode="bilinear", align_corners=False
        )
        anchor = a.permute(0, 2, 3, 1)
    blended = (1.0 - weight) * prev_single + weight * anchor
    return blended.clamp(0.0, 1.0)


def calculate_adaptive_strength(
    current_shot: dict,
    prev_shot: Optional[dict],
    prev_success: bool = True,
) -> float:
    """
    Calculate adaptive anchor strength based on motion/character changes.
    Mirrors PRO v2 logic: high strength for consistency, reduced for flexibility.
    """
    if not USE_ADAPTIVE_STRENGTH:
        return ANCHOR_STRENGTH_HIGH

    strength = ANCHOR_STRENGTH_HIGH

    if prev_shot:
        motion_change = abs(
            current_shot.get("motion_intensity", 0.5)
            - prev_shot.get("motion_intensity", 0.5)
        )
        if motion_change > 0.4:
            strength -= 0.10
        elif motion_change < 0.2:
            strength += 0.05

        if current_shot.get("character_focus") != prev_shot.get("character_focus"):
            strength -= 0.05

    if not prev_success:
        strength -= 0.10

    return max(ANCHOR_STRENGTH_LOW, min(ANCHOR_STRENGTH_HIGH, strength))


# ======================================================================
# Video Helpers
# ======================================================================

def display_video(path: str):
    """Display a video inline in Colab via base64 data URL."""
    if not path or not os.path.exists(path):
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'
    ))


def save_video_obj(video_obj, prefix="IFE") -> str:
    """Save a ComfyUI video object to disk and return the path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        f"video/{prefix}", folder_paths.get_output_directory(), w, h
    )
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(
        path, format=Types.VideoContainer("auto"), codec="auto", metadata=None
    )
    return path


def concatenate_clips(clip_paths: List[str], output_path: str) -> str:
    """ffmpeg concat all clips into one final video."""
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path],
        check=True, capture_output=True,
    )
    print(f"   Concatenated -> {output_path}")
    return output_path


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
            # These failures are typically version mismatches in optional nodes
            # (KJNodes preview, VideoHelperSuite, LTXVideo pyramid blend).
            # Core nodes (UnetLoaderGGUF, DualCLIPLoader, VAELoader, etc.) still work.

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(_l())
        else:
            asyncio.run(_l())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_l())

    # Verify critical nodes are available
    critical_nodes = [
        "UnetLoaderGGUF", "DualCLIPLoader", "CLIPTextEncode",
        "ConditioningZeroOut", "LTXVConditioning", "VAELoader",
        "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent",
        "ManualSigmas", "KSamplerSelect", "RandomNoise", "CFGGuider",
        "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVCropGuides",
        "LTXVLatentUpsampler", "LatentUpscaleModelLoader", "VAEDecode",
        "LTXVAudioVAEDecode", "CreateVideo",
    ]
    missing = [n for n in critical_nodes if n not in NODE_CLASS_MAPPINGS]
    if missing:
        print(f"   [Nodes] WARNING: Missing critical nodes: {missing}")
        print(f"   [Nodes] Available: {len(NODE_CLASS_MAPPINGS)} nodes loaded")
    else:
        print(f"   [Nodes] All {len(critical_nodes)} critical nodes available")

    _NODES_LOADED = True


def _audio_vae(name: str):
    """Load audio VAE using KJ loader if available, else standard loader."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=name, device="main_device", weight_dtype="fp16"
        )
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=name)


# ======================================================================
# LoRA Stack
# ======================================================================

LD_LORA_STACK = [
    {"on": True, "lora": "ltx-2-19b-ic-lora-detailer.safetensors", "guard": False, "strength": 0.4},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
]
_LORA_STACK_JSON = json.dumps(LD_LORA_STACK)


def _apply_loras(unet, clip_model):
    """Apply LD_LORA_STACK to UNet. Tries LTX2MasterLoaderLD first; falls back."""
    active = [
        s for s in LD_LORA_STACK
        if s.get("on") and s.get("lora") not in (None, "None", "")
    ]
    if not active:
        return unet, clip_model
    # LTX2MasterLoaderLD requires a valid clip_model; skip if None
    if clip_model is not None and "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            result = getattr(node, node.FUNCTION)(
                model=unet, clip=clip_model, stack_data=_LORA_STACK_JSON
            )
            print(f"   [LoRA] {len(active)} slot(s) via LTX2MasterLoaderLD")
            return get_value_at_index(result, 0), clip_model
        except Exception as e:
            print(f"   [LoRA] MasterLoader failed ({e}) - manual fallback")
    for slot in active:
        name, strength = slot["lora"], slot["strength"]
        try:
            unet = LoraLoaderModelOnly().load_lora_model_only(unet, name, strength)[0]
            print(f"   [LoRA]  {name} @ {strength}")
        except Exception as e:
            print(f"   [LoRA] skip  {name}: {e}")
    return unet, clip_model


print("Core utilities ready.")



# ======================================================================
# SECTION A -- VISION DESCRIBE ENGINE
# ======================================================================
# @title { "single-column": true }
# @markdown ## Section A - VisionDescribeEngine
# Standalone Qwen2.5-VL wrapper. No ComfyUI node wrapper.
# Loads -> describes -> unloads immediately to free VRAM.


class VisionDescribeEngine:
    """Analyses an image and returns a 100-130 word scene description."""

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }

    PROMPT = (
        "Describe this image in one paragraph of plain sentences, 100-130 words. "
        "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
        "The FIRST sentence about any person MUST explicitly state ethnicity and skin tone "
        "using plain terms: 'a Black man', 'a white woman', 'a South Asian man'. "
        "Include age, hair colour and style, body type, clothing or nude state, pose, "
        "camera framing, angle, lighting, time of day, and setting. "
        "One flowing paragraph, no bullets, no labels. "
        "If no person, describe environment, objects, lighting, mood."
    )

    BIBLE_PROMPT = (
        "Describe this person in structured detail. Provide exactly these attributes:\n"
        "FACE: (face shape, eye color, features)\n"
        "HAIR: (color, style, length)\n"
        "CLOTHING: (all garments, colors, accessories)\n"
        "BUILD: (body type, height estimate)\n"
        "SKIN_TONE: (specific skin tone description)\n"
        "ACCESSORIES: (jewelry, bags, items held)\n"
        "Be precise and specific. Use plain descriptive language."
    )

    def __init__(self, model_key: str = "3B-fast", offline: bool = False):
        self.model_key = model_key
        self.offline = offline

    def _load_model(self):
        """Load model and processor, return (model, processor, process_vision_info)."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from huggingface_hub import snapshot_download
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] pip install qwen-vl-utils")

        hf_id = self.MODEL_OPTIONS[self.model_key]
        if not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [VisionDescribe] Loading {self.model_key} ...")
        processor = AutoProcessor.from_pretrained(source, local_files_only=self.offline)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, device_map="auto", torch_dtype=dtype, local_files_only=self.offline
        )
        model.eval()
        return model, processor, process_vision_info

    def _generate_with_prompt(self, image: Image.Image, prompt_text: str) -> str:
        """Run the VL model with a given prompt and return raw text output."""
        model, processor, process_vision_info = self._load_model()

        messages = [
            {"role": "system", "content":
             "You are an image analysis tool. Describe exactly what you see in plain prose."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(
            text=[text_in], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt"
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        tok = processor.tokenizer
        stop_ids = [i for i in [tok.eos_token_id] if i is not None]
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300, temperature=0.3,
                do_sample=True, top_p=0.9,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=stop_ids,
            )
        result = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()

        del out, inputs, model, processor
        _vram_free()
        return result

    def describe(self, image: Union[Image.Image, torch.Tensor]) -> str:
        """Return a 100-130 word scene description of the image."""
        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)
        desc = self._generate_with_prompt(image, self.PROMPT)
        print(f"   [VisionDescribe] Done  {len(desc.split())} words.")
        return desc

    def describe_for_bible(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, str]:
        """
        Return structured dict with character attributes:
        keys: face, hair, clothing, build, skin_tone, accessories
        """
        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)
        raw = self._generate_with_prompt(image, self.BIBLE_PROMPT)

        # Parse structured output into dict
        result = {
            "face": "", "hair": "", "clothing": "",
            "build": "", "skin_tone": "", "accessories": "",
        }
        current_key = None
        for line in raw.split("\n"):
            line = line.strip()
            for key in result.keys():
                prefix = key.upper() + ":"
                if line.upper().startswith(prefix):
                    current_key = key
                    result[key] = line[len(prefix):].strip()
                    break
            else:
                if current_key and line:
                    result[current_key] += " " + line

        # Fallback: if parsing failed, store raw text in face field
        if not any(result.values()):
            result["face"] = raw

        print(f"   [VisionDescribe] Bible extraction done.")
        return result



# ======================================================================
# SECTION B -- EASY PROMPT ENGINE
# ======================================================================
# @title { "single-column": true }
# @markdown ## Section B - EasyPromptEngine
# Full cinematic LLM expansion with pacing, character bible lock,
# dialogue control, story analysis, camera direction mapping.

# -- Camera Direction Mapping (PRO v2 pattern) --
CAMERA_DIRECTION_MAP = {
    "dolly_forward": "smooth dolly forward pushing into the scene",
    "dolly_backward": "slow dolly backward revealing the wider environment",
    "dolly_left": "lateral dolly sliding left along the scene",
    "dolly_right": "lateral dolly sliding right across the frame",
    "pan_left": "gentle pan rotating left to follow action",
    "pan_right": "gentle pan rotating right to follow action",
    "tilt_up": "slow tilt upward revealing vertical scale",
    "tilt_up_slight": "subtle upward tilt following the subject",
    "tilt_up_dramatic": "dramatic upward tilt emphasizing height and grandeur",
    "tilt_up_reveal": "upward tilt revealing character or environment",
    "tilt_down": "downward tilt from sky to ground level",
    "zoom_in_slow": "gradual zoom tightening the frame on the subject",
    "zoom_in_fast": "rapid zoom punch-in creating urgency",
    "zoom_out": "slow zoom out revealing the wider context",
    "handheld_pov": "slight handheld shake simulating first-person perspective",
    "static_intense": "locked-off static shot building tension through stillness",
    "static_dramatic": "perfectly still dramatic composition",
    "dolly_reveal": "dolly movement revealing a previously hidden element",
    "push_in_slow": "slow push in building intimacy with the subject",
    "low_angle_hero": "low-angle tracking shot giving heroic stature to the subject",
}

# -- Negative prompt builder --
_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)


def _build_neg(result: str, user_input: str) -> str:
    """Build a context-aware negative prompt from the expanded result."""
    c = (result + " " + user_input).lower()
    extras = []
    if any(w in c for w in ["indoor", "room", "interior", "bedroom", "kitchen", "office"]):
        extras.append("harsh outdoor lighting, direct sunlight")
    elif any(w in c for w in ["outdoor", "street", "beach", "forest", "park"]):
        extras.append("studio background, indoor lighting")
    if any(w in c for w in ["close-up", "close up", "portrait", "headshot"]):
        extras.append("wide angle distortion, fish eye")
    elif any(w in c for w in ["wide shot", "wide angle", "aerial"]):
        extras.append("close-up, portrait crop")
    if any(w in c for w in ["night", "dark", "moonlight", "dimly lit", "candlelight"]):
        extras.append("overexposed, bright daylight, blown highlights")
    elif any(w in c for w in ["daylight", "sunny", "golden hour", "bright"]):
        extras.append("underexposed, dark shadows, black crush")
    if any(w in c for w in ["two women", "two men", "two people", "couple", "both"]):
        extras.append("merged bodies, fused figures, incorrect number of people")
    # Character consistency negatives
    extras.append("character morphing, face changing, inconsistent design")
    return ", ".join([_NEG_BASE] + extras)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, then unloads to free VRAM.

    Also provides story analysis workflow:
    - analyze_story(): parse story text into environment + characters
    - extract_characters(): extract character dicts for CharacterBible
    - generate_scene_prompts(): generate per-scene (positive, negative) tuples
    """

    MODELS = {
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = (
        "You are a cinematic prompt writer for LTX-2, an AI video generation model. "
        "Expand the user's idea into a rich, video-ready prompt.\n\n"
        "PRIORITY ORDER:\n"
        "1. Video style & genre (slow-burn thriller, documentary, editorial, action blockbuster)\n"
        "2. Camera angle & shot type (low-angle close-up, bird's-eye wide, Dutch angle medium)\n"
        "3. Character description - age MUST be a specific number (e.g. 'a 28-year-old woman'), "
        "body type, hair, skin, clothing. Use exact words from the user.\n"
        "4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)\n"
        "5. Action & motion - continuous present-tense sequence with clear physical movements.\n"
        "6. Camera movement - describe in prose (dolly forward, tilt up, pan left). "
        "Match movement to the emotional beat of the scene.\n"
        "7. Audio - max 2 ambient sounds active at once, woven as prose.\n\n"
        "MOTION & CAMERA RULES:\n"
        "- Always specify camera motion as prose, never as screenplay brackets.\n"
        "- Match camera energy to action energy (fast action = dynamic camera).\n"
        "- Include at least one specific physical movement per 4 seconds of video.\n"
        "- Describe how characters move through space, not just what they look like.\n\n"
        "RULES:\n"
        "- Present tense throughout.\n"
        "- 8-12 sentences of dense flowing prose - no bullet lists.\n"
        "- Fill the full token budget. Do not stop early.\n"
        "- Output ONLY the expanded prompt. No preamble. No trailing notes."
    )

    STORY_ANALYSIS_PROMPT = (
        "Analyze the following story text and extract structured information. "
        "Return a JSON object with two keys:\n"
        "1. 'environment': an object with keys: location, time, weather, mood, lighting, color_palette\n"
        "2. 'characters': a list of objects, each with keys: name, description, "
        "detailed_appearance (object with: face, hair, clothing, build, skin_tone, accessories)\n\n"
        "Be specific and detailed. Use only information present in the text. "
        "Output ONLY valid JSON, nothing else."
    )

    _CLEAN_RE = [
        (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
        (re.compile(r"^(Sure!?|Certainly!?|Here(?:'s| is).*?:)[^\n]*\n?", re.IGNORECASE), ""),
        (re.compile(r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$", re.IGNORECASE), ""),
        (re.compile(r"\s*\n+Note:.*$", re.DOTALL), ""),
        (re.compile(r"\s*\(Note:.*$", re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\s*(\([^)]{5,120}\)\s*){2,}$", re.DOTALL), ""),
        (re.compile(
            r"\s*\n+(Please let me know|Let me revise|Confirmed\.|Output ends|"
            r"Done\.|I hope|Thank you|No further).*$",
            re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\n{3,}"), "\n\n"),
    ]

    def __init__(self, model_size: str = "8B", offline: bool = False,
                 keep_loaded: bool = False):
        self.model_size = model_size
        self.offline = offline
        self.keep_loaded = keep_loaded
        self._tok = None
        self._model = None
        self._loaded_key = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        key = self.model_size
        if self._model is not None and self._loaded_key == key:
            return
        if self._model is not None:
            self._unload()
        hf_id = self.MODELS[key]
        if not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [EasyPrompt] Loading {key} ...")
        self._tok = AutoTokenizer.from_pretrained(source, local_files_only=self.offline)
        self._model = AutoModelForCausalLM.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            trust_remote_code=True, local_files_only=self.offline
        )
        self._model.config.use_cache = True
        self._model.eval()
        self._loaded_key = key
        print(f"   [EasyPrompt] Loaded.")

    def _unload(self):
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._tok = None
        self._loaded_key = None
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        delims = [
            "assistant", "user", "system", "<|eot_id|>", "<|end_of_turn|>",
            "<|im_end|>", "<end_of_turn>", "[/INST]", "### Human", "### Assistant",
        ]
        ids = [self._tok.eos_token_id]
        for s in delims:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return [i for i in dict.fromkeys(ids) if i is not None]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        for pattern, repl in EasyPromptEngine._CLEAN_RE:
            text = pattern.sub(repl, text)
        text = re.sub(r"\s*[\(\[]\s*$", "", text)
        return text.strip()

    def _raw_generate(self, system_prompt: str, user_content: str,
                      max_tokens: int = 600, temperature: float = 0.9) -> str:
        """Low-level generation: load model, run inference, return raw text."""
        self._load()

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"enable_thinking": False} if is_qwen3 else {}
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **kwargs
        )

        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self._model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self._model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self._model.device)
        else:
            input_ids = raw.to(self._model.device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=temperature, do_sample=True, top_k=40, top_p=0.9,
                repetition_penalty=1.07, use_cache=True,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._stop_ids(),
            )
        result = self._tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del out, input_ids
        return result

    def generate(
        self,
        user_input: str,
        frame_count: int = 121,
        creativity: float = 0.9,
        seed: int = -1,
        scene_context: str = "",
        lora_triggers: str = "",
        character_bible: str = "",
    ) -> Tuple[str, str]:
        """
        Returns (positive_prompt, negative_prompt).

        character_bible is injected as a hard [CHARACTER BIBLE - NON-NEGOTIABLE]
        block so the LLM cannot alter hair, age, clothing, or any locked attribute.
        """
        real_seconds = frame_count / 25.0
        action_count = max(1, min(10, round(real_seconds / 4)))
        token_budget = max(256, min(1200, action_count * 120))
        max_tokens = int(token_budget * 1.05)
        min_tokens = int(token_budget * 0.75)

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Pacing constraint
        ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
        if action_count > 1:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
                f"distinct actions. "
                f"HARD STOP after the {ordinal} action. "
                f"Write ~{token_budget} tokens. No trailing notes or brackets."
            )
        else:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY 1 action. "
                f"HARD STOP after it. ~{token_budget} tokens."
            )

        # Character bible lock
        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                "\n[CHARACTER BIBLE - NON-NEGOTIABLE: Every character attribute below "
                "MUST remain exactly as described. Do NOT alter hair, age, skin, "
                "clothing, or any other attribute. This overrides any inference:\n"
                f"{character_bible.strip()}\n]"
            )

        # Scene context (from Vision Describe)
        if scene_context.strip():
            effective = (
                "[SCENE CONTEXT FROM IMAGE - authoritative, do not contradict]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION - action, style, mood]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        # LoRA trigger injection
        lora_clause = ""
        if lora_triggers.strip():
            lora_clause = f"\n[LORA: Begin prompt with: {lora_triggers.strip()}]"

        user_content = f"{pacing}{bible_clause}{lora_clause}\n\n{effective}"

        self._load()

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"enable_thinking": False} if is_qwen3 else {}
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, **kwargs
        )

        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self._model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self._model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self._model.device)
        else:
            input_ids = raw.to(self._model.device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids, min_new_tokens=min_tokens, max_new_tokens=max_tokens,
                temperature=creativity, do_sample=True, top_k=40, top_p=0.9,
                repetition_penalty=1.07, use_cache=True,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._stop_ids(),
            )

        result = self._tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        result = self._clean(result)
        result = re.sub(r"\s*[\(\[]\s*$", "", result).strip()
        del out, input_ids
        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] Done  {len(result.split())} words generated.")
        return result, neg

    def analyze_story(self, story_text: str) -> Dict[str, Any]:
        """
        Analyze story text and return structured dict with:
        - 'environment': dict with location, time, weather, mood, lighting, color_palette
        - 'characters': list of dicts with name, description, detailed_appearance
        """
        result_text = self._raw_generate(
            system_prompt=self.STORY_ANALYSIS_PROMPT,
            user_content=story_text,
            max_tokens=1200,
            temperature=0.5,
        )
        if not self.keep_loaded:
            self._unload()

        # Try to parse JSON from the response
        try:
            # Find JSON in the response (may be wrapped in markdown code blocks)
            json_match = re.search(r"\{[\s\S]*\}", result_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(result_text)
        except (json.JSONDecodeError, ValueError):
            # Fallback: return minimal structure
            parsed = {
                "environment": {
                    "location": "unspecified",
                    "time": "unspecified",
                    "weather": "clear",
                    "mood": "neutral",
                    "lighting": "natural",
                    "color_palette": "natural tones",
                },
                "characters": [],
            }
            print("   [EasyPrompt] Warning: could not parse story analysis JSON, using fallback.")

        # Ensure expected keys exist
        if "environment" not in parsed:
            parsed["environment"] = {}
        for key in ["location", "time", "weather", "mood", "lighting", "color_palette"]:
            parsed["environment"].setdefault(key, "unspecified")
        if "characters" not in parsed:
            parsed["characters"] = []

        print(f"   [EasyPrompt] Story analyzed: {len(parsed['characters'])} characters found.")
        return parsed

    def extract_characters(self, story_text: str) -> List[Dict[str, Any]]:
        """
        Extract characters from story text as a list of dicts suitable
        for CharacterBible.add_from_json().
        Each dict has: name, description, detailed_appearance (with face, hair, etc.)
        """
        analysis = self.analyze_story(story_text)
        return analysis.get("characters", [])

    def generate_scene_prompts(
        self,
        scenes: List[str],
        bible: "CharacterBible",
        environment: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        """
        Generate per-scene (positive, negative) prompt tuples.
        Injects character bible and environment context into each prompt.
        """
        results = []
        env_context = (
            f"ENVIRONMENT: {environment.get('location', '')}. "
            f"TIME: {environment.get('time', '')}. "
            f"WEATHER: {environment.get('weather', '')}. "
            f"MOOD: {environment.get('mood', '')}. "
            f"LIGHTING: {environment.get('lighting', '')}. "
            f"COLOR PALETTE: {environment.get('color_palette', '')}."
        )
        bible_block = bible.to_prompt_block() if bible.has_characters() else ""

        for scene_beat in scenes:
            scene_input = f"{env_context}\n\n{scene_beat}"
            pos, neg = self.generate(
                user_input=scene_input,
                character_bible=bible_block,
            )
            results.append((pos, neg))

        return results



# ======================================================================
# SECTION C -- CHARACTER BIBLE
# ======================================================================
# @title { "single-column": true }
# @markdown ## Section C - CharacterBible
# Serialisable cross-scene character consistency record.
# Enhanced with PRO v2 features: add_from_json, consistency_prefix,
# lock_attributes, validate_against_description.


class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as character_bible so the
    LLM receives a hard NON-NEGOTIABLE constraint preventing attribute drift.

    Auto-populated from Vision Describe output on the seed image (recommended),
    or filled manually with add() or add_from_json().
    """

    def __init__(self):
        self._chars: Dict[str, dict] = {}
        self._locked: Dict[str, List[str]] = {}

    def add(self, name: str, **attributes):
        """Manually define a character with keyword attributes."""
        self._chars[name] = dict(attributes)

    def add_from_json(self, character_dict: Dict[str, Any]):
        """
        Accept PRO v2 style dict with detailed_appearance.
        Expected keys: name, desc (or description), detailed_appearance
        detailed_appearance should have: face, hair, clothing, build, skin_tone, accessories
        """
        name = character_dict.get("name", "Unknown")
        desc = character_dict.get("desc", character_dict.get("description", ""))
        appearance = character_dict.get("detailed_appearance", {})

        entry = {"description": desc}
        for key in ["face", "hair", "clothing", "build", "skin_tone", "accessories"]:
            if key in appearance:
                entry[key] = appearance[key]

        self._chars[name] = entry

    def extract_from_description(self, name: str, description: str):
        """Store a raw Vision Describe output under a character name."""
        self._chars[name] = {"_raw": description.strip()}

    def consistency_prefix(self) -> str:
        """
        Returns a CHARACTER CONSISTENCY CRITICAL block (PRO v2 pattern).
        This is a strong enforcement prefix for all prompts to maintain
        exact character appearance throughout the entire scene.
        """
        if not self._chars:
            return ""
        char_blocks = []
        for name, attrs in self._chars.items():
            if "_raw" in attrs:
                char_blocks.append(f"{name}: {attrs['_raw'][:200]}")
            else:
                parts = []
                for key in ["face", "hair", "clothing", "build", "skin_tone"]:
                    if key in attrs and attrs[key]:
                        parts.append(attrs[key])
                if parts:
                    char_blocks.append(f"{name}: {', '.join(parts)}")
                elif "description" in attrs:
                    char_blocks.append(f"{name}: {attrs['description'][:200]}")

        combined = " | ".join(char_blocks)
        return (
            "CHARACTER CONSISTENCY CRITICAL: " + combined +
            " | MAINTAIN EXACT SAME CHARACTER APPEARANCE THROUGHOUT ENTIRE SCENE. "
            "NO MORPHING. NO STYLE CHANGES."
        )

    def lock_attributes(self, name: str, attributes_list: List[str]):
        """
        Mark specific attributes as immutable for a character.
        Locked attributes will be flagged during validate_against_description().
        """
        self._locked[name] = list(attributes_list)

    def validate_against_description(self, name: str, description: str) -> List[str]:
        """
        Check a description for drift against locked attributes.
        Returns a list of warnings for any locked attribute not found in the description.
        """
        warnings_list = []
        if name not in self._chars:
            return [f"Character '{name}' not found in bible."]
        locked = self._locked.get(name, [])
        if not locked:
            return []
        char_data = self._chars[name]
        desc_lower = description.lower()
        for attr_key in locked:
            expected_val = char_data.get(attr_key, "")
            if expected_val and isinstance(expected_val, str):
                # Check if key terms from the attribute appear in the description
                key_terms = [t.strip().lower() for t in expected_val.split(",") if len(t.strip()) > 3]
                missing_terms = [t for t in key_terms[:3] if t not in desc_lower]
                if missing_terms:
                    warnings_list.append(
                        f"Drift detected for {name}.{attr_key}: "
                        f"missing terms {missing_terms}"
                    )
        return warnings_list

    def to_prompt_block(self) -> str:
        """Returns the injection string for EasyPromptEngine."""
        if not self._chars:
            return ""
        lines = []
        for name, attrs in self._chars.items():
            if "_raw" in attrs:
                lines.append(f"CHARACTER - {name}:\n{attrs['_raw']}")
            else:
                parts = []
                for k, v in attrs.items():
                    if k != "description" and v:
                        parts.append(f"{k}: {v}")
                if parts:
                    lines.append(f"CHARACTER - {name}: {'; '.join(parts)}")
                elif "description" in attrs:
                    lines.append(f"CHARACTER - {name}: {attrs['description']}")
        return "\n\n".join(lines)

    def has_characters(self) -> bool:
        return bool(self._chars)

    def names(self) -> List[str]:
        return list(self._chars.keys())

    def save(self, path: str):
        data = {"chars": self._chars, "locked": self._locked}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "chars" in data:
            self._chars = data["chars"]
            self._locked = data.get("locked", {})
        else:
            # Legacy format: just the chars dict
            self._chars = data
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"



# ======================================================================
# SECTION D -- GENERATE CLIP (two-pass LTX-2 19B GGUF pipeline)
# ======================================================================
# @title { "single-column": true }
# @markdown ## Section D - generate_clip()
#
# VRAM sequence (T4-safe, peak ~13 GB):
#   1. Load CLIP Gemma -> encode text -> DELETE CLIP -> free
#   2. Load UNet GGUF -> apply LoRAs
#   3. Load VAE_video + VAE_audio + upscaler
#   4. Prepare latents (I2V via LTXVImgToVideoInplace if image, else T2V)
#   5. Pass 1 (ManualSigmas + euler, CFG=1.0)
#   6. TinyVAE Preview (taeltx2_3.safetensors) -> quick decode -> display -> delete
#   7. Pass 2 (spatial upscale + I2V RE-CONDITIONING + gradient_estimation)
#      KEY: Re-apply LTXVImgToVideoInplace after upscale (LD-I2V.json pattern)
#   8. Delete UNet
#   9. Tiled VAE decode (spatial_tiles=2, spatial_overlap=8, temporal_tile_length=48)
#  10. Audio decode
#  11. Save as h264-mp4 at 25fps


def generate_clip(
    image_tensor: Optional[torch.Tensor],
    prompt: str,
    neg_prompt: str,
    width: int = 768,
    height: int = 512,
    frames: int = 121,
    fps: int = 25,
    seed: int = 42,
    image_strength: float = 1.0,
    anchor_image: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.25,
    overlap_frames: int = 0,
    prev_clip_path: Optional[str] = None,
    pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler: str = "euler",
    pass1_cfg: float = 1.0,
    pass2_sigmas: str = "0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler: str = "gradient_estimation",
    pass2_cfg: float = 1.0,
    pass2_seed: int = 0,
    use_tiled_vae: bool = True,
    tiled_stiles: int = 2,
    tiled_soverlap: int = 8,
    tiled_tlen: int = 48,
    tiled_toverlap: int = 4,
    show_preview: bool = True,
    output_prefix: str = "IFE",
) -> str:
    """
    Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is deleted before UNet loads; UNet is deleted before decode.

    Key enhancements:
    - overlap_frames: number of frames for SVI-Pro overlap conditioning
    - prev_clip_path: path to previous clip for extracting overlap frames directly
    - I2V re-conditioning in Pass 2 (LD-I2V.json pattern): after LTXVLatentUpsampler,
      re-apply LTXVImgToVideoInplace with the reference image on the upscaled latent
    - anchor_image: Optional tensor of the original seed image for drift prevention

    anchor_weight: Blend factor for anchor_image (0.0 = pure prev_samples,
        1.0 = pure anchor). Default 0.25 provides subtle consistency.
    """
    _load_comfy_nodes()

    # Overlap frame extraction fallback: only runs if the caller did NOT already
    # provide an image_tensor. When _run_svi_pro_extend passes both overlap_frames
    # and prev_clip_path alongside a pre-extracted image_tensor, this guard
    # (image_tensor is None) correctly prevents double-extraction.
    if prev_clip_path and overlap_frames > 0 and image_tensor is None:
        overlap_batch = extract_overlap_frames(prev_clip_path, n_frames=overlap_frames)
        if overlap_batch is not None:
            # NOTE: Single-frame conditioned - LTXVImgToVideoInplace only accepts
            # single images. For true multi-frame conditioning, latent-space blending
            # of the full overlap batch would be needed.
            image_tensor = overlap_batch[-1:, :, :, :]
            print(f"   [Clip] Extracted {overlap_frames} overlap frames (using last frame for conditioning)")

    img_bypass = image_tensor is None
    img_str = image_strength if not img_bypass else 0.0

    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}x{height}  {frames}f  seed={seed}")
    _vram_print("before clip load")

    with torch.inference_mode():

        # -- STEP 1: CLIP -> encode -> delete --
        print("   [Clip] Loading CLIP...")
        clip_ld = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_raw = get_value_at_index(
            clip_ld.load_clip(
                clip_name1=_M_CLIP1, clip_name2=_M_CLIP2,
                type="ltxv", device="default"
            ), 0
        )

        cte = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos = cte.encode(text=prompt, clip=clip_raw)
        # For CFG > 1.0, encode the negative prompt through CLIP for proper guidance.
        # For CFG=1.0 (default for distilled models), use zeroed-out conditioning
        # since the negative has no effect and encoding it wastes compute.
        if pass1_cfg > 1.0 and neg_prompt:
            cond_neg_raw = cte.encode(text=neg_prompt, clip=clip_raw)
            cond_neg = get_value_at_index(cond_neg_raw, 0)
        else:
            zero_out = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_neg = get_value_at_index(
                zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0)), 0
            )
        ltxv_cn = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond = ltxv_cn.EXECUTE_NORMALIZED(
            frame_rate=float(fps),
            positive=get_value_at_index(cond_pos, 0),
            negative=cond_neg,
        )

        # Delete CLIP now - frees ~6-8 GB for UNet
        del clip_raw, clip_ld, cte
        _vram_free()
        _vram_print("CLIP deleted")

        # -- STEP 2: UNet + LoRAs --
        print("   [Clip] Loading UNet (GGUF Q4_K_M)...")
        unet_ld = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet = get_value_at_index(unet_ld.load_unet(unet_name=_M_UNET), 0)
        _dummy_clip = None
        unet, _ = _apply_loras(unet, _dummy_clip)
        _vram_print("UNet + LoRAs loaded")

        # -- STEP 3: VAEs + upscaler --
        vae_ld = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_v = get_value_at_index(vae_ld.load_vae(vae_name=_M_VAE), 0)
        vae_a = get_value_at_index(_audio_vae(_M_AVAE), 0)
        uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        up_mdl = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=_M_UP), 0)

        # -- STEP 4: Latent preparation --
        ei = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_i = ei.generate(width=width, height=height, batch_size=1, color=0)
        rimn = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_i = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_i, 0), scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5},
        )
        gis = NODE_CLASS_MAPPINGS["GetImageSize"]()
        hsz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_i, 0))
        hw, hh = get_value_at_index(hsz, 0), get_value_at_index(hsz, 1)

        eltxv = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=hw, height=hh, length=frames, batch_size=1
        )

        # I2V: condition the latent on the seed/last-frame image
        # When anchor_image is provided, blend with image_tensor for consistency
        conditioning_img = None
        if not img_bypass:
            conditioning_img = image_tensor
            if anchor_image is not None:
                conditioning_img = blend_anchor_with_prev(
                    anchor=anchor_image, prev_frames=image_tensor, weight=anchor_weight
                )

            rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ri2 = rim2.EXECUTE_NORMALIZED(
                input=conditioning_img, scale_method="lanczos",
                resize_type={
                    "resize_type": "scale dimensions",
                    "width": hw * 2, "height": hh * 2, "crop": "center",
                },
            )
            ppn = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp = get_value_at_index(
                ppn.EXECUTE_NORMALIZED(
                    img_compression=33, image=get_value_at_index(ri2, 0)
                ), 0
            )
            i2v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            vid_lat = (
                get_value_at_index(
                    i2v.EXECUTE_NORMALIZED(
                        strength=img_str, bypass=False,
                        vae=vae_v, image=pp,
                        latent=get_value_at_index(vid_lat, 0),
                    ), 0
                ),
            )
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Audio latent + concat AV
        elalat = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_a
        )
        catav = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat = get_value_at_index(
            catav.EXECUTE_NORMALIZED(
                video_latent=vid_lat[0],
                audio_latent=get_value_at_index(aud_lat, 0),
            ), 0
        )

        # -- STEP 5: Pass 1 --
        print("   [Clip] Pass 1...")
        _vram_print()
        ms = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ks = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        rn = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfg_node = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        g1 = cfg_node.EXECUTE_NORMALIZED(
            cfg=pass1_cfg, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
        )
        o1 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed), 0),
            guider=get_value_at_index(g1, 0),
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass1_sampler), 0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0),
            latent_image=av_lat,
        )
        p1_av = get_value_at_index(o1, 0)
        del g1
        _vram_free()
        print("   [Clip] Pass 1 done")

        # -- STEP 6: TinyVAE Preview --
        if show_preview:
            try:
                print("   [Clip] TinyVAE preview...")
                sep_prev = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
                s_prev = sep_prev.EXECUTE_NORMALIZED(av_latent=p1_av)
                vl_prev = get_value_at_index(s_prev, 0)

                tiny_vae_ld = NODE_CLASS_MAPPINGS["VAELoader"]()
                tiny_vae = get_value_at_index(
                    tiny_vae_ld.load_vae(vae_name=_M_TAEV), 0
                )

                vd_prev = NODE_CLASS_MAPPINGS["VAEDecode"]()
                prev_frames_decoded = get_value_at_index(
                    vd_prev.decode(samples=vl_prev, vae=tiny_vae), 0
                )

                if prev_frames_decoded is not None and prev_frames_decoded.shape[0] > 0:
                    preview_img = tensor_to_pil(prev_frames_decoded[0:1])
                    display(preview_img)
                    print("   [Clip] TinyVAE preview displayed")

                del tiny_vae, tiny_vae_ld, prev_frames_decoded, vl_prev, s_prev, sep_prev, vd_prev
                _vram_free()
            except Exception as e:
                print(f"   [Clip] TinyVAE preview skipped: {e}")

        # -- STEP 7: Pass 2 (upscale + I2V re-conditioning + refine) --
        # This implements the LD-I2V.json differentiator pattern:
        # After LTXVLatentUpsampler, re-apply LTXVImgToVideoInplace with
        # the reference image on the upscaled latent before second sampling.
        print("   [Clip] Pass 2 (upscale + I2V re-condition + refine)...")
        sep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1 = sep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vl1, al1 = get_value_at_index(s1, 0), get_value_at_index(s1, 1)

        crop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cr = crop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vl1,
        )
        g2 = cfg_node.EXECUTE_NORMALIZED(
            cfg=pass2_cfg, model=unet,
            positive=get_value_at_index(cr, 0),
            negative=get_value_at_index(cr, 1),
        )

        # Spatial upscale
        ltxvup = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        up = ltxvup.upsample_latent(
            samples=get_value_at_index(cr, 2),
            upscale_model=up_mdl, vae=vae_v,
        )
        del up_mdl
        _vram_free()

        # LD-I2V.json PATTERN: Re-apply I2V conditioning on upscaled latent
        # This is the key differentiator - it re-conditions the upscaled latent
        # with the reference image for stronger character/scene consistency in Pass 2
        upscaled_latent = get_value_at_index(up, 0)
        if not img_bypass and conditioning_img is not None:
            print("   [Clip] Re-applying I2V conditioning on upscaled latent (LD-I2V pattern)...")
            # Delete existing VAE before re-conditioning to avoid duplicate VAE in VRAM.
            # Matches reference PRO v2 single-VAE-at-a-time pattern.
            del vae_v, vae_ld
            _vram_free()

            # Load fresh VAE for I2V re-conditioning
            vae_ld2 = NODE_CLASS_MAPPINGS["VAELoader"]()
            vae_v2 = get_value_at_index(vae_ld2.load_vae(vae_name=_M_VAE), 0)

            # Resize image to match upscaled dimensions
            rim3 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ri3 = rim3.EXECUTE_NORMALIZED(
                input=conditioning_img, scale_method="lanczos",
                resize_type={
                    "resize_type": "scale dimensions",
                    "width": width, "height": height, "crop": "center",
                },
            )
            ppn2 = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp2 = get_value_at_index(
                ppn2.EXECUTE_NORMALIZED(
                    img_compression=33, image=get_value_at_index(ri3, 0)
                ), 0
            )
            i2v_pass2 = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            i2v_result = i2v_pass2.EXECUTE_NORMALIZED(
                strength=img_str, bypass=False,
                vae=vae_v2, image=pp2,
                latent=upscaled_latent,
            )
            upscaled_latent = get_value_at_index(i2v_result, 0)
            del vae_v2, vae_ld2
            _vram_free()
            print("   [Clip] I2V re-conditioning applied.")

        # Concat upscaled video + audio for Pass 2
        av2 = get_value_at_index(
            catav.EXECUTE_NORMALIZED(video_latent=upscaled_latent, audio_latent=al1), 0
        )

        o2 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=pass2_seed), 0),
            guider=get_value_at_index(g2, 0),
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass2_sampler), 0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0),
            latent_image=av2,
        )
        p2_den = get_value_at_index(o2, 1)

        # BIG VRAM RELEASE - UNet is done
        del g2, unet, unet_ld
        _vram_free()
        _vram_print("UNet deleted")
        print("   [Clip] Pass 2 done")

        # Reload VAE for decode if it was deleted during I2V re-conditioning
        if not img_bypass and conditioning_img is not None:
            vae_ld = NODE_CLASS_MAPPINGS["VAELoader"]()
            vae_v = get_value_at_index(vae_ld.load_vae(vae_name=_M_VAE), 0)

        # -- STEP 8: Decode video --
        s2 = sep.EXECUTE_NORMALIZED(av_latent=p2_den)
        vl_f = get_value_at_index(s2, 0)
        al_f = get_value_at_index(s2, 1)

        decoded = None
        if use_tiled_vae:
            try:
                td = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded = get_value_at_index(
                    td.EXECUTE_NORMALIZED(
                        vae=vae_v, latents=vl_f,
                        spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                        temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                        last_frame_fix=False, working_device="auto", working_dtype="auto",
                    ), 0
                )
                print("   [Clip] Tiled VAE done")
            except (KeyError, Exception) as e:
                print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}) - standard decode")
                use_tiled_vae = False

        if not use_tiled_vae:
            vd = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = get_value_at_index(vd.decode(samples=vl_f, vae=vae_v), 0)

        del vae_v
        _vram_free()

        # -- STEP 9: Audio decode --
        aud_d = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)
        del vae_a
        _vram_free()

        # -- STEP 10: Save --
        cv_node = NODE_CLASS_MAPPINGS["CreateVideo"]()
        vid_obj = cv_node.EXECUTE_NORMALIZED(
            fps=fps, images=decoded, audio=get_value_at_index(audio, 0)
        )
        path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=output_prefix)
        _vram_print("after save")
        return path



# ======================================================================
# SECTION E -- INFINITE FLOW ENGINE
# ======================================================================
# @title { "single-column": true }
# @markdown ## Section E - InfiniteFlowEngine
# Orchestrator supporting three modes:
#   - scene_chain: separate clips per story beat, chained via last-frame
#   - temporal_extend: initial clip extended via last-frame I2V
#   - svi_pro_extend: proper SVI-Pro extension with overlap frames,
#     anchor_samples, adaptive strength, and overlap-aware stitching


class InfiniteFlowEngine:
    """
    Generates video sequences using LTX-2 19B GGUF with three operational modes:

    MODE 1 - scene_chain (default):
        Generates separate clips per story beat, chains via last-frame extraction.

    MODE 2 - temporal_extend:
        Generates an initial clip, then extends it multiple times using last-frame
        as conditioning. Maintains anchor_samples throughout for consistency.

    MODE 3 - svi_pro_extend (NEW):
        Proper SVI-Pro extension mode:
        - Extracts last OVERLAP_FRAMES frames from each segment (not just last frame)
        - Uses those frames for conditioning the next segment
        - Maintains anchor_samples throughout all extensions
        - Adaptive anchor strength calculation based on motion/character changes
        - Overlap-aware video stitching (trims overlap from joined clips)

    Every heavyweight model is loaded -> used -> unloaded in strict sequence.
    Peak VRAM on T4: ~13 GB (UNet + VAEs + upscaler during Pass 2).
    """

    def __init__(
        self,
        character_bible: Optional["CharacterBible"] = None,
        llm_model: str = "8B",
        vision_model: str = "3B-fast",
        width: int = 768,
        height: int = 512,
        frames: int = 121,
        fps: int = 25,
        image_strength: float = 1.0,
        creativity: float = 0.9,
        lora_triggers: str = "",
        use_vision: bool = True,
        use_tiled_vae: bool = True,
        show_previews: bool = True,
        offline: bool = False,
        output_dir: str = "/content/ComfyUI/output/infinite_flow",
        mode: str = "scene_chain",
        extensions_per_clip: int = 3,
    ):
        self.bible = character_bible or CharacterBible()
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.width = width
        self.height = height
        self.frames = frames
        self.fps = fps
        self.img_strength = image_strength
        self.creativity = creativity
        self.lora_triggers = lora_triggers
        self.use_vision = use_vision
        self.use_tiled_vae = use_tiled_vae
        self.show_previews = show_previews
        self.offline = offline
        self.output_dir = output_dir
        self.mode = mode
        self.extensions_per_clip = extensions_per_clip
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._vision_engine = VisionDescribeEngine(vision_model, offline)
        self._prompt_engine = EasyPromptEngine(llm_model, offline, keep_loaded=False)

        self._clip_paths: List[str] = []
        self._scene_history: List[str] = []

    # -- Rolling context (forward continuity) --
    def _rolling_ctx(self) -> str:
        return "\n".join(self._scene_history[-2:])

    # -- Single beat processing --
    def _process_beat(self, beat: str, current_image: Optional[torch.Tensor],
                      beat_idx: int, base_seed: int) -> Tuple[str, str, int]:
        beat_seed = base_seed + (beat_idx - 1) * 1000

        # Vision Describe - only from beat 2 onward
        if beat_idx > 1 and current_image is not None and self.use_vision:
            print(f"   [Beat {beat_idx}] Vision Describe...")
            scene_ctx = self._vision_engine.describe(current_image)
            _vram_free()
        else:
            scene_ctx = self._rolling_ctx()

        # Expand beat -> cinematic prompt with character bible lock
        print(f"   [Beat {beat_idx}] EasyPrompt expand...")
        prompt, neg = self._prompt_engine.generate(
            user_input=beat,
            frame_count=self.frames,
            creativity=self.creativity,
            seed=beat_seed,
            scene_context=scene_ctx,
            lora_triggers=self.lora_triggers,
            character_bible=self.bible.to_prompt_block(),
        )
        _vram_free()

        # Store for next beat rolling context
        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        return prompt, neg, beat_seed

    # -- Scene Chain mode --
    def _run_scene_chain(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """Generate separate clips per story beat, chain via last-frame extraction."""
        current_image: Optional[torch.Tensor] = None

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image...")
                desc = self._vision_engine.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)

        for beat_idx, beat in enumerate(story_beats, 1):
            print(f"\n[IFE] -- Beat {beat_idx}/{len(story_beats)} --")
            print(f"[IFE]   {beat[:80]}{'...' if len(beat) > 80 else ''}")

            prompt, neg, beat_seed = self._process_beat(
                beat, current_image, beat_idx, base_seed
            )
            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}...")

            try:
                clip_path = generate_clip(
                    image_tensor=current_image,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=beat_seed,
                    image_strength=self.img_strength,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=self.show_previews,
                    output_prefix=f"IFE_{beat_idx:03d}",
                )
            except torch.cuda.OutOfMemoryError:
                _vram_free()
                reduced_frames = max(61, self.frames // 2)
                print(f"   OOM on beat {beat_idx}. Retrying with {reduced_frames} frames...")
                clip_path = generate_clip(
                    image_tensor=None,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=reduced_frames,
                    fps=self.fps,
                    seed=beat_seed + 1,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=False,
                    output_prefix=f"IFE_{beat_idx:03d}_retry",
                )

            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            print(f"\n[IFE] Beat {beat_idx} -> {dest}")

            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else None

            if self.show_previews:
                display_video(dest)

        return self._clip_paths

    # -- Temporal Extend mode --
    def _run_temporal_extend(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Generate an initial clip then extend via last-frame I2V conditioning.
        Maintains anchor_samples (original seed image) for visual consistency.
        """
        current_image: Optional[torch.Tensor] = None
        anchor_samples: Optional[torch.Tensor] = None

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            anchor_samples = current_image.clone()
            print(f"[IFE-TE] Seed/anchor image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                desc = self._vision_engine.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)

        total_segments = 1 + self.extensions_per_clip
        print(f"\n[IFE-TE] TEMPORAL EXTEND MODE: {total_segments} segments")

        for seg_idx in range(total_segments):
            is_initial = (seg_idx == 0)
            seg_seed = base_seed + seg_idx * 1000

            beat_idx = min(seg_idx, len(story_beats) - 1)
            beat = story_beats[beat_idx]
            if seg_idx >= len(story_beats):
                beat = f"Continue seamlessly from the previous scene. {beat}"

            print(f"\n[IFE-TE] -- Segment {seg_idx + 1}/{total_segments} --")

            if not is_initial and current_image is not None and self.use_vision:
                scene_ctx = self._vision_engine.describe(current_image)
                _vram_free()
            else:
                scene_ctx = self._rolling_ctx()

            prompt, neg = self._prompt_engine.generate(
                user_input=beat,
                frame_count=self.frames,
                creativity=self.creativity,
                seed=seg_seed,
                scene_context=scene_ctx,
                lora_triggers=self.lora_triggers,
                character_bible=self.bible.to_prompt_block(),
            )
            _vram_free()
            self._scene_history.append(prompt[:400])
            if len(self._scene_history) > 2:
                self._scene_history = self._scene_history[-2:]

            if is_initial:
                conditioning_image = current_image
                clip_anchor = None
            else:
                conditioning_image = current_image
                clip_anchor = anchor_samples

            try:
                clip_path = generate_clip(
                    image_tensor=conditioning_image,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=seg_seed,
                    image_strength=self.img_strength,
                    anchor_image=clip_anchor,
                    anchor_weight=0.25,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=self.show_previews,
                    output_prefix=f"IFE_TE_{seg_idx + 1:03d}",
                )
            except torch.cuda.OutOfMemoryError:
                _vram_free()
                reduced_frames = max(61, self.frames // 2)
                clip_path = generate_clip(
                    image_tensor=None,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=reduced_frames,
                    fps=self.fps,
                    seed=seg_seed + 1,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=False,
                    output_prefix=f"IFE_TE_{seg_idx + 1:03d}_retry",
                )

            dest = os.path.join(self.output_dir, f"segment_{seg_idx + 1:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)

            last_frame = get_last_frame_tensor(dest)
            if last_frame is not None:
                current_image = last_frame
            else:
                current_image = anchor_samples

            if self.show_previews:
                display_video(dest)

        return self._clip_paths

    # -- SVI-Pro Extend mode (NEW) --
    def _run_svi_pro_extend(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Proper SVI-Pro temporal extension mode:
        - Extract last OVERLAP_FRAMES frames from each segment
        - Use those frames for conditioning the next segment
        - Maintain anchor_samples throughout all extensions
        - Adaptive anchor strength calculation
        - Overlap-aware video stitching (trim overlap from joined clips)
        """
        current_image: Optional[torch.Tensor] = None
        anchor_samples: Optional[torch.Tensor] = None
        prev_clip_path: Optional[str] = None
        prev_shot_data: Optional[dict] = None
        prev_success: bool = True

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            anchor_samples = current_image.clone()
            print(f"[IFE-SVI] Anchor/seed image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                desc = self._vision_engine.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)

        total_segments = 1 + self.extensions_per_clip
        print(f"\n[IFE-SVI] SVI-PRO EXTEND MODE")
        print(f"[IFE-SVI]  Overlap frames: {OVERLAP_FRAMES}")
        print(f"[IFE-SVI]  Segments: {total_segments}")
        print(f"[IFE-SVI]  Adaptive strength: {USE_ADAPTIVE_STRENGTH}")

        for seg_idx in range(total_segments):
            is_initial = (seg_idx == 0)
            seg_seed = base_seed + seg_idx * 1000

            beat_idx = min(seg_idx, len(story_beats) - 1)
            beat = story_beats[beat_idx]
            if seg_idx >= len(story_beats):
                beat = f"Continue seamlessly from the previous scene. {beat}"

            # Build shot data for adaptive strength.
            # Vary motion_intensity by segment index: later segments get higher
            # intensity to allow more divergence from the anchor, mimicking
            # natural escalation in multi-segment generation.
            current_shot_data = {
                "motion_intensity": min(0.5 + seg_idx * 0.08, 0.9),
                "character_focus": "main",
                "segment_index": seg_idx,
            }

            # Calculate adaptive anchor strength
            anchor_strength = calculate_adaptive_strength(
                current_shot_data, prev_shot_data, prev_success
            )

            print(f"\n[IFE-SVI] -- Segment {seg_idx + 1}/{total_segments} --")
            print(f"[IFE-SVI]   Anchor strength: {anchor_strength:.2f}")
            print(f"[IFE-SVI]   Seed: {seg_seed}")

            # For non-initial segments, extract overlap frames from previous clip
            overlap_batch = None
            if not is_initial and prev_clip_path:
                overlap_batch = extract_overlap_frames(prev_clip_path, n_frames=OVERLAP_FRAMES)
                if overlap_batch is not None:
                    # NOTE: Current implementation is single-frame conditioned because
                    # LTXVImgToVideoInplace only accepts a single image tensor.
                    # For true multi-frame SVI-Pro conditioning (temporal coherence at
                    # boundaries), a latent-space blending approach would be needed:
                    # encode the overlap batch through VAE into a partial latent,
                    # then blend/concatenate with the new segment's latent before sampling.
                    current_image = overlap_batch[-1:, :, :, :]
                    print(f"[IFE-SVI]   Extracted {OVERLAP_FRAMES} overlap frames (using last frame for conditioning)")

            # Vision describe
            if not is_initial and current_image is not None and self.use_vision:
                scene_ctx = self._vision_engine.describe(current_image)
                _vram_free()
            else:
                scene_ctx = self._rolling_ctx()

            # Prompt generation
            prompt, neg = self._prompt_engine.generate(
                user_input=beat,
                frame_count=self.frames,
                creativity=self.creativity,
                seed=seg_seed,
                scene_context=scene_ctx,
                lora_triggers=self.lora_triggers,
                character_bible=self.bible.to_prompt_block(),
            )
            _vram_free()
            self._scene_history.append(prompt[:400])
            if len(self._scene_history) > 2:
                self._scene_history = self._scene_history[-2:]

            # Determine conditioning for generate_clip
            if is_initial:
                clip_image = current_image
                clip_anchor = None
                clip_anchor_weight = 0.0
            else:
                clip_image = current_image
                clip_anchor = anchor_samples
                clip_anchor_weight = anchor_strength

            try:
                clip_path = generate_clip(
                    image_tensor=clip_image,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=seg_seed,
                    image_strength=self.img_strength,
                    anchor_image=clip_anchor,
                    anchor_weight=clip_anchor_weight,
                    overlap_frames=OVERLAP_FRAMES if not is_initial else 0,
                    prev_clip_path=prev_clip_path if not is_initial else None,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=self.show_previews,
                    output_prefix=f"IFE_SVI_{seg_idx + 1:03d}",
                )
                prev_success = True
            except torch.cuda.OutOfMemoryError:
                _vram_free()
                reduced_frames = max(61, self.frames // 2)
                print(f"   OOM on segment {seg_idx + 1}. Retrying with {reduced_frames} frames...")
                clip_path = generate_clip(
                    image_tensor=None,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=reduced_frames,
                    fps=self.fps,
                    seed=seg_seed + 1,
                    use_tiled_vae=self.use_tiled_vae,
                    show_preview=False,
                    output_prefix=f"IFE_SVI_{seg_idx + 1:03d}_retry",
                )
                prev_success = False

            dest = os.path.join(self.output_dir, f"svi_segment_{seg_idx + 1:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            prev_clip_path = dest
            prev_shot_data = current_shot_data

            print(f"[IFE-SVI] Segment {seg_idx + 1} -> {dest}")

            if self.show_previews:
                display_video(dest)

        print(f"\n[IFE-SVI] {total_segments} segments complete.")
        return self._clip_paths

    # -- Story-to-prompt workflow --
    def run_from_story(
        self,
        story_text: str,
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Full story-to-prompt pipeline:
        1. Call EasyPromptEngine.analyze_story() for environment + characters
        2. Build CharacterBible from extracted characters
        3. Generate per-scene prompts
        4. Generate clips using the configured mode
        """
        print("[IFE] === STORY-TO-PROMPT WORKFLOW ===")

        # Step 1: Analyze story
        print("[IFE] Step 1: Analyzing story text...")
        analysis = self._prompt_engine.analyze_story(story_text)
        environment = analysis.get("environment", {})
        characters = analysis.get("characters", [])

        # Step 2: Build CharacterBible
        print(f"[IFE] Step 2: Building CharacterBible ({len(characters)} characters)...")
        for char_dict in characters:
            self.bible.add_from_json(char_dict)
        if self.bible.has_characters():
            print(f"[IFE]   Bible: {self.bible.names()}")

        # Step 3: Split story into scene beats (use paragraphs or sentences)
        scene_beats = [
            s.strip() for s in story_text.split("\n") if s.strip() and len(s.strip()) > 20
        ]
        if not scene_beats:
            scene_beats = [story_text]

        # Step 4: Generate scene prompts
        print(f"[IFE] Step 3: Generating prompts for {len(scene_beats)} scenes...")
        scene_prompts = self._prompt_engine.generate_scene_prompts(
            scene_beats, self.bible, environment
        )

        # Step 5: Generate clips using beats and prompts
        # Convert prompt tuples back to simple beats for the run method
        # (the prompts are already expanded, so we pass them as-is)
        print(f"[IFE] Step 4: Generating clips in '{self.mode}' mode...")
        expanded_beats = [pos for pos, neg in scene_prompts]

        return self.run(
            story_beats=expanded_beats,
            seed_image_path=seed_image_path,
            base_seed=base_seed,
        )

    # -- Main run method --
    def run(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Process story beats according to the configured mode.
        Returns list of output .mp4 paths.
        """
        print(f"\n[IFE] === INFINITE FLOW ENGINE ===")
        print(f"[IFE]  Mode     : {self.mode}")
        print(f"[IFE]  Beats    : {len(story_beats)}")
        print(f"[IFE]  Size     : {self.width}x{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[IFE]  LLM      : {self.llm_model}  Vision: {self.vision_model}")
        print(f"[IFE]  Bible    : {self.bible.names() or 'empty'}")
        if self.mode in ("temporal_extend", "svi_pro_extend"):
            print(f"[IFE]  Extensions: {self.extensions_per_clip}")
        if self.mode == "svi_pro_extend":
            print(f"[IFE]  Overlap   : {OVERLAP_FRAMES} frames")

        if self.mode == "svi_pro_extend":
            return self._run_svi_pro_extend(story_beats, seed_image_path, base_seed)
        elif self.mode == "temporal_extend":
            return self._run_temporal_extend(story_beats, seed_image_path, base_seed)
        else:
            return self._run_scene_chain(story_beats, seed_image_path, base_seed)

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        """Concatenate all generated clips into one final video."""
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)

    def concat_with_overlap_trim(self, output_name: str = "full_video_trimmed.mp4") -> str:
        """
        Concatenate clips with overlap trimming for SVI-Pro mode.
        Trims OVERLAP_FRAMES worth of frames from the beginning of each
        non-first clip to avoid duplicate content at boundaries.
        """
        if not self._clip_paths:
            return ""
        trim_seconds = OVERLAP_FRAMES / float(self.fps)
        list_file = "/tmp/concat_trim_list.txt"
        trimmed_paths = []
        for i, clip_path in enumerate(self._clip_paths):
            if i == 0:
                trimmed_paths.append(clip_path)
            else:
                # Check clip duration before trimming to avoid empty output
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "csv=p=0", clip_path],
                        capture_output=True, text=True, check=True,
                    )
                    clip_duration = float(probe.stdout.strip())
                except (subprocess.CalledProcessError, ValueError):
                    clip_duration = 0.0

                if clip_duration <= trim_seconds:
                    # Clip is too short to trim; use as-is to avoid empty segment
                    print(f"   [Concat] Clip {i} too short ({clip_duration:.2f}s) to trim, using full clip")
                    trimmed_paths.append(clip_path)
                else:
                    # Trim the first overlap_frames from non-first clips
                    trimmed = clip_path.replace(".mp4", "_trimmed.mp4")
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", f"{trim_seconds:.3f}", "-i", clip_path,
                         "-c", "copy", trimmed],
                        check=True, capture_output=True,
                    )
                    trimmed_paths.append(trimmed)

        with open(list_file, "w") as f:
            for p in trimmed_paths:
                f.write(f"file '{p}'\n")
        out = os.path.join(self.output_dir, output_name)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", out],
            check=True, capture_output=True,
        )
        print(f"   Concatenated (overlap-trimmed) -> {out}")
        return out

    def download_all(self):
        """Offer all clips for download in Colab."""
        if files is None:
            print("   [Download] google.colab.files not available outside Colab")
            return
        for p in self._clip_paths:
            if os.path.exists(p):
                files.download(p)

    def save_bible(self, path: str = "/content/character_bible.json"):
        self.bible.save(path)



# ======================================================================
# VoiceSyncHook -- Placeholder for future voice/audio sync integration
# ======================================================================


class VoiceSyncHook:
    """
    Foundation class for future voice synchronization and audio integration.

    Planned integrations:
    - Wav2Lip: lip sync overlay on generated video
    - SadTalker: face animation driven by audio
    - Whisper: automatic speech recognition for timing alignment
    - TTS engines: text-to-speech for dialogue generation

    This class provides the interface that will be implemented when
    audio sync tools are integrated into the pipeline. Currently all
    methods are placeholders that return None or pass-through values.
    """

    def __init__(self, enabled: bool = False, strength: float = 0.95):
        self.enabled = enabled
        self.strength = strength
        self._audio_cache: Dict[str, str] = {}

    def attach_audio(self, video_path: str, audio_path: str) -> Optional[str]:
        """
        Attach an audio track to a video file.

        TODO: Use ffmpeg to mux audio onto video with proper sync.
        Will support: WAV, MP3, AAC input formats.
        Will handle sample rate conversion and duration matching.
        Returns path to the output video with audio, or None on failure.
        """
        # TODO: Implement ffmpeg-based audio attachment
        # subprocess.run(["ffmpeg", "-i", video_path, "-i", audio_path,
        #                "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
        print("[VoiceSyncHook] attach_audio: not yet implemented")
        return None

    def generate_lipsync(self, video_path: str, transcript: str) -> Optional[str]:
        """
        Generate lip-synced video from transcript.

        TODO: Integrate Wav2Lip or SadTalker for lip sync generation.
        Pipeline:
        1. Generate TTS audio from transcript
        2. Extract face regions from video
        3. Apply lip sync model to match audio
        4. Composite results back onto original video
        Returns path to lip-synced video, or None if not available.
        """
        # TODO: Implement Wav2Lip/SadTalker pipeline
        print("[VoiceSyncHook] generate_lipsync: not yet implemented")
        return None

    def align_dialogue(
        self, shots: List[Dict[str, Any]], dialogue_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Align dialogue entries to shot timings.

        TODO: Match dialogue entries to their corresponding shots based on
        time codes. Adjust timing for natural speech rhythm.
        Returns shots list with dialogue_audio paths attached.
        """
        # TODO: Implement timing alignment logic
        # For each dialogue entry, find the corresponding shot by time range,
        # generate TTS audio, and attach the path to the shot dict.
        print("[VoiceSyncHook] align_dialogue: not yet implemented")
        return shots

    def get_audio_timing(self, dialogue_entry: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate audio timing parameters for a dialogue entry.

        TODO: Analyze dialogue text to estimate:
        - duration: estimated speech duration based on word count and language
        - start_offset: delay before speech begins within the shot
        - fade_in: audio fade-in duration
        - fade_out: audio fade-out duration
        Returns dict with timing parameters.
        """
        # TODO: Implement duration estimation based on word count
        # Rough estimate: ~150 words per minute for natural speech
        text = dialogue_entry.get("dialogue", "")
        word_count = len(text.split())
        estimated_duration = word_count / 2.5  # ~150 wpm = 2.5 words/sec
        print("[VoiceSyncHook] get_audio_timing: using rough estimation")
        return {
            "duration": estimated_duration,
            "start_offset": 0.0,
            "fade_in": 0.1,
            "fade_out": 0.1,
        }


print("All sections defined.")
print("   A: VisionDescribeEngine  B: EasyPromptEngine  C: CharacterBible")
print("   D: generate_clip()       E: InfiniteFlowEngine")
print("   +: VoiceSyncHook (placeholder)")



# ======================================================================
# CELL 4 -- CONFIGURATION (edit before running Cell 5)
# ======================================================================
# @title { "single-column": true }
# @markdown ## 4. Configuration

# -- LLM / Vision --
LLM_MODEL = "3B"          # @param ["8B","3B","14B"]
VISION_MODEL = "3B-fast"  # @param ["3B-fast","7B-nsfw"]
CREATIVITY = 0.9          # @param {type:"number"}

# -- Video settings --
WIDTH = 768           # @param {type:"integer"}
HEIGHT = 512          # @param {type:"integer"}
FRAMES = 121          # @param {type:"integer"}
FPS = 25              # @param {type:"integer"}
BASE_SEED = 42        # @param {type:"integer"}
IMAGE_STRENGTH = 1.0  # @param {type:"number"}
USE_TILED_VAE = True  # @param {type:"boolean"}
USE_VISION = True     # @param {type:"boolean"}
SHOW_PREVIEWS = True  # @param {type:"boolean"}

# -- SVI-Pro Settings --
# OVERLAP_FRAMES = 16       (defined in Cell 3 constants)
# ANCHOR_STRENGTH_HIGH = 0.85
# ANCHOR_STRENGTH_LOW = 0.70
# USE_ADAPTIVE_STRENGTH = True

# -- Reference / seed image --
SEED_IMAGE_PATH = None  # @param {type:"string"}

# -- LoRA stack --
LD_LORA_STACK[0] = {
    "on": True,
    "lora": "ltx-2-19b-ic-lora-detailer.safetensors",
    "guard": False,
    "strength": 0.4,
}
_LORA_STACK_JSON = json.dumps(LD_LORA_STACK)

# -- Generation Mode --
# Options: "scene_chain", "temporal_extend", "svi_pro_extend"
GENERATION_MODE = "temporal_extend"  # @param ["scene_chain","temporal_extend","svi_pro_extend"]
EXTENSIONS_PER_CLIP = 3             # @param {type:"integer"}

# -- Character Bible (manual override or leave empty for auto-extraction) --
bible = CharacterBible()
# Example manual entry (comment out to use auto-extraction):
# bible.add("Elena",
#     age=28, ethnicity="white British", hair="long auburn waves",
#     eyes="green", build="slender athletic",
#     clothes="cream linen blouse, dark skinny jeans, white trainers",
#     other="small scar above left eyebrow, silver hoop earrings")

# -- Story Text (alternative to STORY_BEATS - for run_from_story workflow) --
# Set STORY_TEXT to a multiline string for the full story analysis pipeline.
# When STORY_TEXT is provided and STORY_BEATS is empty, run_from_story() is used.
STORY_TEXT = ""  # @param {type:"string"}

# -- Story Beats (traditional per-beat input) --
STORY_BEATS = [
    ("Elena arrives at the entrance of a dimly lit urban apartment building at night, "
     "pushing through the glass door, rain dripping from her jacket"),

    ("She takes the elevator, watching the floor numbers tick upward, "
     "her reflection ghostly in the steel doors"),

    ("Elena unlocks her front door and steps inside the dark apartment, "
     "not turning on the lights, setting her keys on the counter quietly"),

    ("She moves to the kitchen, opens the fridge, stares blankly at the shelves, "
     "the cold blue light illuminating her face"),

    ("Elena notices a note on the kitchen table that was not there this morning, "
     "she picks it up slowly, her expression shifting from curiosity to alarm"),

    ("She grabs her phone from her bag, dials a number, presses it to her ear - "
     "no answer. She tries again. Silence. She lowers the phone."),

    ("Elena walks to the window and peers down at the wet street below, "
     "watching a black car parked at the curb with its engine running"),

    ("She goes to the bedroom wardrobe, pulls out a small travel bag, "
     "begins packing quickly - clothes, passport, a laptop"),

    ("A sharp knock at the front door. Elena freezes mid-motion, "
     "bag half-packed, listening. Silence. Then another knock, louder."),

    ("She approaches the door, looks through the peephole - "
     "a man in a grey coat stands in the corridor, face turned away"),

    ("Elena slips out through the apartment service exit, "
     "moving quickly down the back stairwell, bag over her shoulder"),

    ("She emerges onto the rain-slicked alley behind the building, "
     "looks both ways, then runs toward the far end where a taxi waits with its light on"),
]

print("Configuration ready.")
print(f"   Mode     : {GENERATION_MODE}")
print(f"   Beats    : {len(STORY_BEATS)}")
print(f"   Video    : {WIDTH}x{HEIGHT}  {FRAMES}f @ {FPS}fps  (~{FRAMES / FPS:.1f}s per clip)")
print(f"   LLM      : {LLM_MODEL}  Vision: {VISION_MODEL}  Creativity: {CREATIVITY}")
print(f"   Seed img : {SEED_IMAGE_PATH or 'None (T2V mode)'}")
print(f"   Bible    : {bible or 'auto-extract from seed image'}")
print(f"   Overlap  : {OVERLAP_FRAMES} frames")
if GENERATION_MODE in ("temporal_extend", "svi_pro_extend"):
    print(f"   Extensions: {EXTENSIONS_PER_CLIP}")
if STORY_TEXT:
    print(f"   Story text: {len(STORY_TEXT)} chars (will use run_from_story)")



# ======================================================================
# CELL 5 -- RUN
# ======================================================================
# @title { "single-column": true }
# @markdown ## 5. Run Infinite Flow Engine


def _run():
    """Main execution entry point."""

    # Decide whether to use story text workflow or traditional beats
    use_story_workflow = bool(STORY_TEXT.strip()) and not STORY_BEATS

    engine = InfiniteFlowEngine(
        character_bible=bible,
        llm_model=LLM_MODEL,
        vision_model=VISION_MODEL,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        fps=FPS,
        image_strength=IMAGE_STRENGTH,
        creativity=CREATIVITY,
        lora_triggers="",
        use_vision=USE_VISION,
        use_tiled_vae=USE_TILED_VAE,
        show_previews=SHOW_PREVIEWS,
        offline=False,
        output_dir="/content/ComfyUI/output/infinite_flow",
        mode=GENERATION_MODE,
        extensions_per_clip=EXTENSIONS_PER_CLIP,
    )

    if use_story_workflow:
        # Full story analysis pipeline
        print("[Run] Using STORY_TEXT workflow (analyze -> bible -> prompts -> clips)")
        clip_paths = engine.run_from_story(
            story_text=STORY_TEXT,
            seed_image_path=SEED_IMAGE_PATH,
            base_seed=BASE_SEED,
        )
    else:
        # Traditional per-beat workflow
        clip_paths = engine.run(
            story_beats=STORY_BEATS,
            seed_image_path=SEED_IMAGE_PATH,
            base_seed=BASE_SEED,
        )

    # Save character bible for reuse across sessions
    engine.save_bible("/content/character_bible.json")

    # Concatenate clips into one final video
    if len(clip_paths) > 1:
        if GENERATION_MODE == "svi_pro_extend":
            final = engine.concat_with_overlap_trim("svi_pro_final.mp4")
        elif GENERATION_MODE == "temporal_extend":
            final = engine.concat_all("extended_final.mp4")
        else:
            final = engine.concat_all("scene_chain_final.mp4")
        print(f"\n Final video: {final}")
        if SHOW_PREVIEWS:
            display_video(final)

    # Offer download
    engine.download_all()
    return engine


_run()
