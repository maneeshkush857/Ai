#!/usr/bin/env python3
"""
LTX 2.3 Director 2.0 Music Video (30s) - Complete Python Pipeline
=================================================================
Converted from: "1. LTX 2.3_Director_2.0 MV Workflow 30s 260802-2.json"
Reference patterns from: "Experiment_LTX23_TI2V_Distilled.ipynb.txt"

This script generates a 30-second cinematic music video at 1280x720, 24fps (756 frames)
using the LTX 2.3 Director 2.0 workflow with a 2-stage sampling pipeline:
  Stage 1: 8 steps, denoise 1.0 (initial generation at lower resolution)
  Stage 2: 4 steps, denoise 0.42 (upscaled refinement)

Designed for Google Colab with GPU (A100/L4 recommended).
"""

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ==============================================================================

# --- pip installs ---
import subprocess
import sys
import os

def install_packages():
    """Install all required Python packages."""
    packages = [
        "torch", "torchvision", "torchaudio",
        "torchsde", "einops", "diffusers", "accelerate",
        "av", "spandrel", "albumentum", "onnx", "opencv-python", "onnxruntime",
        "nest_asyncio"
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + packages,
        check=True
    )
    print("Python packages installed.")

# Uncomment the following line when running in Colab:
# install_packages()

# --- Clone ComfyUI and custom nodes ---
def clone_repos():
    """Clone ComfyUI and all required custom nodes."""
    os.chdir("/content")

    # Clone main ComfyUI
    if not os.path.exists("/content/ComfyUI"):
        subprocess.run(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r",
             "/content/ComfyUI/requirements.txt"],
            check=True
        )

    custom_nodes_dir = "/content/ComfyUI/custom_nodes"
    os.makedirs(custom_nodes_dir, exist_ok=True)
    os.chdir(custom_nodes_dir)

    repos = {
        "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes",
        "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF",
        "ComfyUI-LTXVideo": "https://github.com/Lightricks/ComfyUI-LTXVideo",
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "whatdreamscost-comfyui": "https://github.com/whatdreamscost/whatdreamscost-comfyui",
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy",
    }

    for name, url in repos.items():
        path = os.path.join(custom_nodes_dir, name)
        if not os.path.exists(path):
            subprocess.run(["git", "clone", url], check=True)
        req_file = os.path.join(path, "requirements.txt")
        if os.path.exists(req_file):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
                check=True
            )

    os.chdir("/content/ComfyUI")
    print("All repositories cloned and requirements installed.")

# Uncomment the following line when running in Colab:
# clone_repos()

# --- Install system packages ---
def install_system_packages():
    """Install aria2 and ffmpeg via apt."""
    subprocess.run(
        ["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
        check=True, capture_output=True
    )
    print("System packages (aria2, ffmpeg) installed.")

# Uncomment the following line when running in Colab:
# install_system_packages()

# --- Set up sys.path and imports ---
sys.path.insert(0, "/content/ComfyUI")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import json
import numpy as np
from pathlib import Path
from typing import Sequence, Mapping, Any, Union


# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.
    Some return a dictionary, in these cases, we look for the "results" key.

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def download_with_aria2c(link, folder="/content/ComfyUI/models/loras"):
    """Download a file using aria2c with 16 connections."""
    filename = link.split("/")[-1].split("?")[0]
    os.makedirs(folder, exist_ok=True)
    command = (
        f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "
        f"{link} -d {folder} -o {filename}"
    )
    print(f"Downloading {filename}...")
    os.system(command)
    return filename


def model_download(url: str, dest_dir: str, filename: str = None, silent: bool = True) -> str:
    """
    Colab-optimized download with aria2c.

    Args:
        url: Download URL
        dest_dir: Target directory (will be created if needed)
        filename: Optional output filename (defaults to URL filename)
        silent: If True, suppresses all output (except errors)

    Returns:
        str: filename if successful, False if failed
    """
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split("/")[-1].split("?")[0]

        cmd = [
            "aria2c",
            "--console-log-level=error",
            "-c", "-x", "16", "-s", "16", "-k", "1M",
            "-d", dest_dir,
            "-o", filename,
            url
        ]

        if silent:
            cmd.extend(["--summary-interval=0", "--quiet"])
            print(f"Downloading {filename}...", end=" ", flush=True)

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if silent:
            print("Done!")
        else:
            print(f"Downloaded {filename} to {dest_dir}")
        return filename

    except subprocess.CalledProcessError as e:
        error = e.stderr.strip() or "Unknown error"
        print(f"\nError downloading {filename}: {error}")
        return False
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False


def import_custom_nodes() -> None:
    """Load all built-in and external custom nodes in a Jupyter/Colab-safe way."""
    import asyncio
    import nest_asyncio

    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def loader():
        import_failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if import_failed:
            print("WARNING: Some comfy_extras nodes failed to import:")
            for node in import_failed:
                print(" -", node)

    try:
        asyncio.run(loader())
    except RuntimeError:
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(loader())


def display_video(video_path):
    """Display a video in Colab notebook."""
    from IPython.display import HTML, display
    from base64 import b64encode

    video_data = open(video_path, "rb").read()
    if video_path.lower().endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path.lower().endswith(".webm"):
        mime_type = "video/webm"
    else:
        mime_type = "video/mp4"

    data_url = f"data:{mime_type};base64," + b64encode(video_data).decode()
    display(HTML(f"""
    <video width=768 controls autoplay loop>
        <source src="{data_url}" type="{mime_type}">
    </video>
    """))


def clear_memory():
    """Aggressive GPU and CPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def upload_file():
    """Handle file upload in Colab and return the path."""
    import shutil
    from google.colab import files as colab_files

    os.makedirs("/content/ComfyUI/input", exist_ok=True)
    uploaded = colab_files.upload()
    paths = []
    for filename in uploaded.keys():
        src_path = f"/content/ComfyUI/{filename}"
        dest_path = f"/content/ComfyUI/input/{filename}"
        shutil.move(src_path, dest_path)
        paths.append(dest_path)
        print(f"File saved to: {dest_path}")
    return paths[0] if paths else None



# ==============================================================================
# SECTION 3: MODEL DOWNLOADS
# ==============================================================================

def download_all_models():
    """Download all required models for the LTX Director 2.0 MV pipeline."""

    print("=" * 60)
    print("DOWNLOADING ALL MODELS")
    print("=" * 60)

    # --- DiT / UNET (GGUF quantized) ---
    # Node 135: UnetLoaderGGUF
    model_download(
        "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/ltx-2-3-22b-dev-Q4_K_M.gguf",
        "/content/ComfyUI/models/unet"
    )

    # --- Text Encoders ---
    # Node 12: DualCLIPLoader - clip_name1
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/gemma_3_12B_it_fp4_mixed.safetensors",
        "/content/ComfyUI/models/text_encoders"
    )
    # Node 12: DualCLIPLoader - clip_name2
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/ltx-2.3_text_projection_bf16.safetensors",
        "/content/ComfyUI/models/text_encoders"
    )

    # --- VAE Models ---
    # Node 36: VAELoader - Video VAE
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/LTX23_video_vae_bf16.safetensors",
        "/content/ComfyUI/models/vae"
    )
    # Node 8: VAELoader - Audio VAE
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/LTX23_audio_vae_bf16.safetensors",
        "/content/ComfyUI/models/vae"
    )
    # Node 6: VAELoaderKJ - Tiny VAE for preview
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/taeltx2_3.safetensors",
        "/content/ComfyUI/models/vae"
    )

    # --- Latent Upscale Model ---
    # Node 13: LatentUpscaleModelLoader
    model_download(
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "/content/ComfyUI/models/latent_upscale_models"
    )

    # --- LoRA Models ---
    # Node 138: Power Lora Loader (rgthree) - 4 LoRAs
    # LoRA 1: Distilled (strength 0.4)
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
        "/content/ComfyUI/models/loras"
    )
    # LoRA 2: OmniNFT-RL (strength 0.6)
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "/content/ComfyUI/models/loras"
    )
    # LoRA 3: Transition (strength 0.7)
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/ltx2.3-transition.safetensors",
        "/content/ComfyUI/models/loras"
    )
    # LoRA 4: MVCamera-drclips (strength 0.9)
    model_download(
        "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main/LTX2.3-MVCamera-drclips.safetensors",
        "/content/ComfyUI/models/loras"
    )

    print("=" * 60)
    print("ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("=" * 60)


# Uncomment to download models:
# download_all_models()


# ==============================================================================
# SECTION 4: MAIN GENERATION FUNCTION DEFINITION
# ==============================================================================

# The full global prompt from the JSON workflow (LTXDirector node 131 properties)
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
- Perform with explosive stage presence.
- Every musical phrase immediately creates a new emotional and physical performance.
- Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, hands, posture and body rhythm.
- The performance continuously builds toward emotional peaks.
- Own the stage with absolute confidence.
- Perform as if in front of 50,000 screaming fans.
- Captivate the audience every second.
- Never appear calm, passive or static.

Facial Performance:
- Extremely expressive facial acting throughout the entire performance.
- Rich emotional transitions every few words.
- Powerful eye contact with intense emotional engagement.
- Eyes sparkle with confidence and passion.
- Highly expressive eyebrows synchronized with important lyrics.
- Strong cheek and jaw movement while singing.
- Natural smiles, smirks, determination, excitement, confidence, attitude, passion, curiosity, joy and intensity.
- Rich cinematic micro-expressions.
- Never hold the same facial expression for more than a brief musical phrase.
- The face should feel emotionally alive every second.

Body Performance:
- The entire body constantly grooves with the beat.
- Strong rhythmic bouncing.
- Powerful shoulder accents.
- Confident chest movement.
- Hip movement follows the groove.
- Frequent body turns.
- Fast weight shifts.
- Dynamic torso twists.
- Lean toward the camera during emotional lyrics.
- Occasionally step toward the camera.
- Performance intensity increases naturally during powerful musical moments.
- Bold, energetic and theatrical stage movement.

Hand Performance:
- Perform like an experienced pop or hip-hop superstar.
- Large expressive gestures.
- Fast rhythmic arm accents.
- Sharp hand movements synchronized with the beat.
- Powerful pointing.
- Sweeping arm movements.
- Punching the air.
- Pulling gestures toward the chest.
- Throwing gestures outward.
- Finger snapping.
- Open palm emphasis.
- Framing the face.
- Expressive wrist movement.
- Hands constantly create visual rhythm.
- One hand naturally leads while the other follows.
- Asymmetrical movement.
- Avoid symmetrical gestures.
- Never repeatedly raise both hands together.
- Every musical phrase introduces fresh gestures.
- Never repeat the same gesture pattern.

Musical Timing:
- Body movement follows musical phrasing rather than every word.
- Strong beats create explosive movements.
- Soft phrases become intimate and emotional.
- Fast lyrics generate faster gestures.
- Slow lyrics become smoother without losing energy.
- Every movement feels rhythmically connected to the music.

Speech Synchronization:
- Perfect lip synchronization.
- Accurate mouth shapes.
- Expressions and gestures match the emotional meaning of every lyric.
- Natural breathing between phrases.

Motion Quality:
- Premium AI human animation.
- Fast, confident and energetic performance.
- Realistic momentum.
- Strong acceleration and deceleration.
- High-energy body mechanics.
- Natural motion blur.
- No robotic movement.
- No frozen poses.
- No repetitive gesture loops.
- No presenter-style gestures.
- No idle standing.
- No jitter.
- No flickering.
- No facial distortion.
- No identity drift.
- No hand deformation.
- No extra fingers.
- No malformed limbs.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. Camera movement follows the beat and amplifies the performance.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, every second feels alive, impossible to look away.

Spoken dialogue:
"Open up the canvas, blank space on my screen. 
Drag a Checkpoint Loader, you know what I mean.
KSampler in the middle, VAE on the right,
Put the Text Encoder, yeah, building tonight.
Connect the nodes, run the queue,
Watch the latent flow right through.
Green, nothing green, nothing yellow,
Positive Prompt, in my hub."
"""


def main_ltx_director_mv(
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    duration_seconds: float = 31.5,
    duration_frames: int = 756,
    global_prompt: str = GLOBAL_PROMPT,
    image_paths: list = None,
    audio_file_path: str = "whatdreamscost/Late night trap.mp3",
):
    """
    Main LTX Director 2.0 Music Video generation function.
    
    Implements the full 2-stage pipeline from the JSON workflow:
      - Stage 1: 8 steps, denoise=1.0, guide_weight=0.5 (initial low-res generation)
      - Stage 2: 4 steps, denoise=0.42, guide_weight=1.0 (upscaled high-res refinement)
    
    Args:
        seed: Random seed for reproducibility (node 30)
        width: Output video width (1280)
        height: Output video height (720)
        fps: Frame rate (24)
        duration_seconds: Video duration in seconds (31.5)
        duration_frames: Total frame count (756)
        global_prompt: The cinematic prompt for the LTXDirector node
        image_paths: List of 5 image paths for keyframe segments
        audio_file_path: Path to the audio file (relative to ComfyUI input)
    """

    # Default image paths matching the JSON workflow segments
    if image_paths is None:
        image_paths = [
            "whatdreamscost/1.png",
            "whatdreamscost/2.png",
            "whatdreamscost/3.png",
            "whatdreamscost/4.png",
            "whatdreamscost/5.3.png",
        ]

    # Load custom nodes (async-safe for Colab)
    import_custom_nodes()

    from nodes import NODE_CLASS_MAPPINGS

    print("=" * 60)
    print("LTX 2.3 DIRECTOR 2.0 - MUSIC VIDEO GENERATION")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Duration: {duration_seconds}s")
    print(f"Total Frames: {duration_frames} | Seed: {seed}")
    print("=" * 60)

    # ==========================================================================
    # SECTION 5: MODEL LOADING
    # ==========================================================================

    with torch.inference_mode():

        # ----------------------------------------------------------------------
        # 5a) Load GGUF Model (Node 135: UnetLoaderGGUF)
        # ----------------------------------------------------------------------
        print("\n[1/7] Loading GGUF DiT model...")
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_135 = unetloadergguf.load_unet(
            unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"
        )

        # ----------------------------------------------------------------------
        # 5b) Load CLIP with DualCLIPLoader (Node 12)
        # Gemma 3 12B fp4 + LTX 2.3 text projection
        # ----------------------------------------------------------------------
        print("[2/7] Loading DualCLIP text encoders...")
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_12 = dualcliploader.load_clip(
            clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
            clip_name2="ltx-2.3_text_projection_bf16.safetensors",
            type="ltxv",
            device="default",
        )

        # ----------------------------------------------------------------------
        # 5c) Apply 4 LoRAs (Node 138: Power Lora Loader rgthree)
        # Strengths: 0.4, 0.6, 0.7, 0.9
        # ----------------------------------------------------------------------
        print("[3/7] Loading LoRAs...")

        # Get the model and clip from previous nodes
        model_from_unet = get_value_at_index(unetloadergguf_135, 0)
        clip_from_dual = get_value_at_index(dualcliploader_12, 0)

        # Try Power Lora Loader (rgthree) first, fallback to sequential LoraLoaderModelOnly
        lora_configs = [
            ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", 0.4),
            ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.6),
            ("ltx2.3-transition.safetensors", 0.7),
            ("LTX2.3-MVCamera-drclips.safetensors", 0.9),
        ]

        if "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS:
            # Use Power Lora Loader (rgthree) - Node 138
            print("  Using Power Lora Loader (rgthree)...")
            power_lora_loader = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()

            # Build the lora_stack configuration for Power Lora Loader
            # The rgthree Power Lora Loader takes model, clip, and lora configurations
            power_lora_result = power_lora_loader.load_loras(
                model=model_from_unet,
                clip=clip_from_dual,
                lora_1=lora_configs[0][0],
                strength_1=lora_configs[0][1],
                lora_2=lora_configs[1][0],
                strength_2=lora_configs[1][1],
                lora_3=lora_configs[2][0],
                strength_3=lora_configs[2][1],
                lora_4=lora_configs[3][0],
                strength_4=lora_configs[3][1],
            )
            model_with_loras = get_value_at_index(power_lora_result, 0)
            clip_with_loras = get_value_at_index(power_lora_result, 1)
        else:
            # Fallback: Sequential LoraLoaderModelOnly
            print("  Power Lora Loader not found. Using sequential LoraLoaderModelOnly...")
            from nodes import LoraLoaderModelOnly

            current_model = model_from_unet
            for lora_name, strength in lora_configs:
                loader = LoraLoaderModelOnly()
                current_model = loader.load_lora_model_only(
                    current_model, lora_name, strength
                )[0]
                print(f"    Loaded: {lora_name} (strength={strength})")

            model_with_loras = current_model
            clip_with_loras = clip_from_dual

        print(f"  All 4 LoRAs applied successfully.")

        # Memory cleanup after LoRA loading
        del unetloadergguf_135
        clear_memory()

        # ----------------------------------------------------------------------
        # 5d) ModelPreviewOverrideKJ (Node 10) with tiny VAE
        # Uses taeltx2_3.safetensors for fast preview, preview_width=240, fps=24
        # ----------------------------------------------------------------------
        print("[4/7] Setting up model preview override...")

        # Load tiny VAE (Node 6: VAELoaderKJ)
        if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            tiny_vae_result = vaeloaderkj.load_vae(
                vae_name="taeltx2_3.safetensors",
                device="main_device",
                weight_dtype="bf16"
            )
        else:
            vaeloader_generic = NODE_CLASS_MAPPINGS["VAELoader"]()
            tiny_vae_result = vaeloader_generic.load_vae(
                vae_name="taeltx2_3.safetensors"
            )
        tiny_vae = get_value_at_index(tiny_vae_result, 0)

        # ModelPreviewOverrideKJ (Node 10)
        modelpreviewoverridekj = NODE_CLASS_MAPPINGS["ModelPreviewOverrideKJ"]()
        modelpreview_10 = modelpreviewoverridekj.EXECUTE_NORMALIZED(
            model=model_with_loras,
            vae=tiny_vae,
            start_index=0,
            max_fps=80,
            autoplay=True,
            preview_width=240,
            fps=24,
            preview_text=""
        )
        model_for_director = get_value_at_index(modelpreview_10, 0)

        # ----------------------------------------------------------------------
        # 5e) VAELoader for Audio VAE (Node 8)
        # LTX23_audio_vae_bf16.safetensors
        # ----------------------------------------------------------------------
        print("[5/7] Loading Audio VAE...")
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_8 = vaeloader.load_vae(
            vae_name="LTX23_audio_vae_bf16.safetensors"
        )
        audio_vae = get_value_at_index(vaeloader_8, 0)

        # ----------------------------------------------------------------------
        # 5f) VAELoader for Video VAE (Node 36)
        # LTX23_video_vae_bf16.safetensors
        # ----------------------------------------------------------------------
        print("[6/7] Loading Video VAE...")
        vaeloader_36 = vaeloader.load_vae(
            vae_name="LTX23_video_vae_bf16.safetensors"
        )
        video_vae = get_value_at_index(vaeloader_36, 0)

        # ----------------------------------------------------------------------
        # 5g) LatentUpscaleModelLoader (Node 13)
        # ltx-2.3-spatial-upscaler-x2-1.1.safetensors
        # ----------------------------------------------------------------------
        print("[7/7] Loading Latent Upscale Model...")
        latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        latentupscalemodelloader_13 = latentupscalemodelloader.EXECUTE_NORMALIZED(
            model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        upscale_model = get_value_at_index(latentupscalemodelloader_13, 0)

        print("\nAll models loaded successfully!")
        print("=" * 60)


        # ==================================================================
        # SECTION 6: LTXDirector NODE (Node 131)
        # ==================================================================
        # The LTXDirector orchestrates the entire music video generation:
        # - Takes model from ModelPreviewOverrideKJ (node 10)
        # - Takes clip from Power Lora Loader (node 138, output 1)
        # - Takes audio_vae from Audio VAE loader (node 8)
        # - Configures timeline with 5 image segments and audio
        # - Outputs: model, positive, video_latent, audio_latent,
        #            guide_data, motion_guide_data, frame_rate
        # ==================================================================

        print("\n[SECTION 6] Calling LTXDirector node...")

        # Build timeline_data JSON matching the workflow exactly
        timeline_data = json.dumps({
            "mainTrackEnabled": True,
            "audioTrackEnabled": True,
            "motionTrackEnabled": True,
            "propHeight": 90,
            "globalPropHeight": 470,
            "showFilenames": True,
            "overrideAudio": False,
            "inpaint_audio": True,
            "global_prompt": global_prompt,
            "retake_global_prompt": "",
            "retakeMode": False,
            "retakeStart": 24,
            "retakeLength": 48,
            "retakePrompt": "",
            "retakeStrength": 1,
            "retakeVideo": None,
            "normalStartFrame": 0,
            "normalDurationFrames": duration_frames,
            "segments": [
                {
                    "id": "1785555235678s2fn3",
                    "start": 0,
                    "length": 226.01059340956584,
                    "prompt": "",
                    "type": "image",
                    "imageFile": image_paths[0],
                    "imageB64": f"/api/view?filename={os.path.basename(image_paths[0])}&type=input&subfolder=whatdreamscost",
                    "isEndFrame": False
                },
                {
                    "id": "17855552413529uw9r",
                    "start": 226.01059340956584,
                    "length": 161.31859976617454,
                    "prompt": "",
                    "type": "image",
                    "imageFile": image_paths[1],
                    "imageB64": f"/api/view?filename={os.path.basename(image_paths[1])}&type=input&subfolder=whatdreamscost",
                    "isEndFrame": False
                },
                {
                    "id": "1785555243885y3h85",
                    "start": 387.3291931757404,
                    "length": 131.45629831196658,
                    "prompt": "",
                    "type": "image",
                    "imageFile": image_paths[2],
                    "imageB64": f"/api/view?filename={os.path.basename(image_paths[2])}&type=input&subfolder=whatdreamscost",
                    "isEndFrame": False
                },
                {
                    "id": "1785555247117rcoma",
                    "start": 518.785491487707,
                    "length": 225.5063328766255,
                    "prompt": "",
                    "type": "image",
                    "imageFile": image_paths[3],
                    "imageB64": f"/api/view?filename={os.path.basename(image_paths[3])}&type=input&subfolder=whatdreamscost",
                    "isEndFrame": False
                },
                {
                    "id": "17855554543736wlrg",
                    "start": 744.2918243643325,
                    "length": 83.22765271847516,
                    "prompt": "",
                    "type": "image",
                    "imageFile": image_paths[4],
                    "imageB64": f"/api/view?filename={os.path.basename(image_paths[4])}&type=input&subfolder=whatdreamscost",
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
                    "audioFile": audio_file_path,
                    "fileName": os.path.basename(audio_file_path)
                }
            ]
        })

        # Instantiate and call LTXDirector (Node 131)
        ltxdirector = NODE_CLASS_MAPPINGS["LTXDirector"]()
        ltxdirector_131 = ltxdirector.EXECUTE_NORMALIZED(
            model=model_for_director,
            clip=clip_with_loras,
            audio_vae=audio_vae,
            optional_latent=None,
            global_prompt=global_prompt,
            start_second=0,
            end_second=duration_seconds,
            duration_seconds=duration_seconds,
            start_frame=0,
            end_frame=duration_frames,
            duration_frames=duration_frames,
            timeline_data=timeline_data,
            local_prompts=" |  |  |  | ",
            segment_lengths="226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,11.708175635667544",
            epsilon=0.001,
            guide_strength="1.00,1.00,1.00,1.00,1.00",
            mainTrackEnabled=True,
            audioTrackEnabled=True,
            motionTrackEnabled=True,
            frame_rate=fps,
            display_mode="seconds",
            custom_width=width,
            custom_height=height,
            resize_method="maintain aspect ratio",
            divisible_by=32,
            img_compression=18,
            retakeMode=False,
            retakeVideo=""
        )

        # Extract LTXDirector outputs (matching node 131 output indices)
        director_model = get_value_at_index(ltxdirector_131, 0)       # output 0: model
        director_positive = get_value_at_index(ltxdirector_131, 1)    # output 1: positive conditioning
        director_video_latent = get_value_at_index(ltxdirector_131, 2)  # output 2: video_latent
        director_audio_latent = get_value_at_index(ltxdirector_131, 3)  # output 3: audio_latent
        director_guide_data = get_value_at_index(ltxdirector_131, 4)   # output 4: guide_data
        director_motion_guide = get_value_at_index(ltxdirector_131, 5)  # output 5: motion_guide_data
        director_frame_rate = get_value_at_index(ltxdirector_131, 6)   # output 6: frame_rate

        print("LTXDirector node executed successfully.")
        print(f"  Frame rate from Director: {director_frame_rate}")


        # ==================================================================
        # SECTION 7: STAGE 1 SAMPLING (8 steps, denoise=1.0)
        # ==================================================================
        # Stage 1 generates the initial video at lower resolution.
        # Flow: ConditioningZeroOut -> LTXVConditioning -> LTXDirectorGuide (133)
        #       -> LTXVConcatAVLatent -> CFGGuider -> SamplerCustomAdvanced (31)
        # ==================================================================

        print("\n[SECTION 7] STAGE 1 - Initial Generation (8 steps, denoise=1.0)")
        print("-" * 60)

        # 7a) ConditioningZeroOut (Node 128)
        # Creates negative conditioning from the director's positive
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningzeroout_128 = conditioningzeroout.zero_out(
            conditioning=director_positive
        )
        negative_cond = get_value_at_index(conditioningzeroout_128, 0)

        # 7b) LTXVConditioning (Node 27)
        # Combines positive from Director, negative from ZeroOut, frame_rate from Director
        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        ltxvconditioning_27 = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=director_frame_rate,
            positive=director_positive,
            negative=negative_cond,
        )
        stage1_positive = get_value_at_index(ltxvconditioning_27, 0)
        stage1_negative = get_value_at_index(ltxvconditioning_27, 1)

        # 7c) LTXDirectorGuide - Stage 1 (Node 133)
        # Inputs: positive/negative from LTXVConditioning, vae=video_vae,
        #         latent=video_latent from Director, guide_data, motion_guide_data, model from Director
        # Widgets: scale_factor=1, guide_weight=0.5, interpolation="bicubic"
        print("  Running LTXDirectorGuide (Stage 1, node 133)...")
        ltxdirectorguide = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
        ltxdirectorguide_133 = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=stage1_positive,
            negative=stage1_negative,
            vae=video_vae,
            latent=director_video_latent,
            guide_data=director_guide_data,
            motion_guide_data=director_motion_guide,
            model=director_model,
            preview_model="None",
            scale_factor=1,
            guide_weight=0.5,
            interpolation="bicubic",
            scale_mode=1,
            crop="center",
            autoplay=True,
            loop=False,
            preview_width=256,
            max_fps=64,
            show_on_node=False,
        )

        # Extract outputs from LTXDirectorGuide Stage 1 (node 133)
        s1_guide_positive = get_value_at_index(ltxdirectorguide_133, 0)   # positive
        s1_guide_negative = get_value_at_index(ltxdirectorguide_133, 1)   # negative
        s1_guide_latent = get_value_at_index(ltxdirectorguide_133, 2)     # latent (video)
        s1_guide_model = get_value_at_index(ltxdirectorguide_133, 3)      # model

        # 7d) LTXVConcatAVLatent (Node 29)
        # Combines video_latent from Stage 1 Guide + audio_latent from Director
        ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        ltxvconcatavlatent_29 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=s1_guide_latent,
            audio_latent=director_audio_latent,
        )
        s1_combined_latent = get_value_at_index(ltxvconcatavlatent_29, 0)

        # 7e) CFGGuider (Node 28) - cfg=1
        # model from Stage 1 Guide, positive/negative from Stage 1 Guide
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        cfgguider_28 = cfgguider.EXECUTE_NORMALIZED(
            cfg=1,
            model=s1_guide_model,
            positive=s1_guide_positive,
            negative=s1_guide_negative,
        )
        s1_guider = get_value_at_index(cfgguider_28, 0)

        # 7f) KSamplerSelect (Node 32) - "euler"
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_32 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
        s1_sampler = get_value_at_index(ksamplerselect_32, 0)

        # 7g) BasicScheduler (Node 33) - linear_quadratic, 8 steps, denoise=1.0
        # model from Stage 1 LTXDirectorGuide (node 133, output 3)
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        basicscheduler_33 = basicscheduler.EXECUTE_NORMALIZED(
            model=s1_guide_model,
            scheduler="linear_quadratic",
            steps=8,
            denoise=1.0,
        )
        s1_sigmas = get_value_at_index(basicscheduler_33, 0)

        # 7h) RandomNoise (Node 30) - seed=0, fixed
        # This noise is reused in Stage 2 as well
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_30 = randomnoise.EXECUTE_NORMALIZED(
            noise_seed=seed
        )
        shared_noise = get_value_at_index(randomnoise_30, 0)

        # 7i) SamplerCustomAdvanced (Node 31) - Stage 1 sampling
        print("  Running Stage 1 SamplerCustomAdvanced (node 31)...")
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        samplercustomadvanced_31 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=shared_noise,
            guider=s1_guider,
            sampler=s1_sampler,
            sigmas=s1_sigmas,
            latent_image=s1_combined_latent,
        )
        s1_output = get_value_at_index(samplercustomadvanced_31, 0)

        print("  Stage 1 sampling complete!")

        # Memory cleanup after Stage 1 sampling
        del cfgguider_28, s1_guider, s1_sigmas
        clear_memory()


        # ==================================================================
        # SECTION 8: STAGE 1 POST-PROCESSING
        # ==================================================================
        # Separate the AV latent from Stage 1, then crop guides for Stage 2.
        # Flow: LTXVSeparateAVLatent (34) -> LTXDirectorCropGuides (55)
        # ==================================================================

        print("\n[SECTION 8] Stage 1 Post-Processing")
        print("-" * 60)

        # 8a) LTXVSeparateAVLatent (Node 34)
        # Separates the combined output from Stage 1 into video and audio latents
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        ltxvseparateavlatent_34 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=s1_output
        )
        s1_video_latent = get_value_at_index(ltxvseparateavlatent_34, 0)  # video_latent
        s1_audio_latent = get_value_at_index(ltxvseparateavlatent_34, 1)  # audio_latent

        # 8b) LTXDirectorCropGuides (Node 55)
        # Inputs: positive/negative from Stage 1 LTXDirectorGuide (node 133 outputs 0,1)
        #         latent = video_latent from node 34 (output 0)
        # Outputs: positive (0), negative (1), latent (2) - for Stage 2 input
        ltxdirectorcropguides = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
        ltxdirectorcropguides_55 = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s1_guide_positive,
            negative=s1_guide_negative,
            latent=s1_video_latent,
        )
        crop55_positive = get_value_at_index(ltxdirectorcropguides_55, 0)
        crop55_negative = get_value_at_index(ltxdirectorcropguides_55, 1)
        crop55_latent = get_value_at_index(ltxdirectorcropguides_55, 2)

        print("  Stage 1 post-processing complete. Ready for Stage 2 upscale.")

        # Memory cleanup between stages
        del s1_output, s1_guide_latent, s1_combined_latent
        clear_memory()


        # ==================================================================
        # SECTION 9: STAGE 2 UPSCALE SAMPLING (4 steps, denoise=0.42)
        # ==================================================================
        # Stage 2 upscales the latent and refines at higher resolution.
        # Flow: LTXVLatentUpsampler (14) -> LTXDirectorGuide (132)
        #       -> LTXVConcatAVLatent (18) -> CFGGuider (17)
        #       -> SamplerCustomAdvanced (19)
        # ==================================================================

        print("\n[SECTION 9] STAGE 2 - Upscale & Refinement (4 steps, denoise=0.42)")
        print("-" * 60)

        # 9a) LTXVLatentUpsampler (Node 14)
        # Upscales the cropped latent from node 55 (output 2)
        # Uses upscale_model from node 13 and video_vae from node 36
        print("  Running LTXVLatentUpsampler (node 14)...")
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        ltxvlatentupsampler_14 = ltxvlatentupsampler.upsample_latent(
            samples=crop55_latent,
            upscale_model=upscale_model,
            vae=video_vae,
        )
        upsampled_latent = get_value_at_index(ltxvlatentupsampler_14, 0)

        # 9b) LTXDirectorGuide - Stage 2 (Node 132)
        # Inputs: positive/negative from CropGuides (node 55, outputs 0,1)
        #         vae=video_vae, latent from upsampler (node 14)
        #         guide_data, motion_guide_data from LTXDirector (node 131)
        #         model from LTXDirector (node 131, output 0)
        # Widgets: scale_factor=1, guide_weight=1.0, interpolation="bicubic"
        print("  Running LTXDirectorGuide (Stage 2, node 132)...")
        ltxdirectorguide_132 = ltxdirectorguide.EXECUTE_NORMALIZED(
            positive=crop55_positive,
            negative=crop55_negative,
            vae=video_vae,
            latent=upsampled_latent,
            guide_data=director_guide_data,
            motion_guide_data=director_motion_guide,
            model=director_model,
            preview_model="None",
            scale_factor=1,
            guide_weight=1.0,
            interpolation="bicubic",
            scale_mode=1,
            crop="center",
            autoplay=True,
            loop=False,
            preview_width=256,
            max_fps=64,
            show_on_node=False,
        )

        # Extract outputs from LTXDirectorGuide Stage 2 (node 132)
        s2_guide_positive = get_value_at_index(ltxdirectorguide_132, 0)   # positive
        s2_guide_negative = get_value_at_index(ltxdirectorguide_132, 1)   # negative
        s2_guide_latent = get_value_at_index(ltxdirectorguide_132, 2)     # latent (video)
        s2_guide_model = get_value_at_index(ltxdirectorguide_132, 3)      # model

        # 9c) LTXVConcatAVLatent (Node 18)
        # Combines video_latent from Stage 2 Guide + audio_latent from Stage 1 separation (node 34, output 1)
        ltxvconcatavlatent_18 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=s2_guide_latent,
            audio_latent=s1_audio_latent,
        )
        s2_combined_latent = get_value_at_index(ltxvconcatavlatent_18, 0)

        # 9d) CFGGuider (Node 17) - cfg=1
        # model from Stage 2 Guide, positive/negative from Stage 2 Guide
        cfgguider_17 = cfgguider.EXECUTE_NORMALIZED(
            cfg=1,
            model=s2_guide_model,
            positive=s2_guide_positive,
            negative=s2_guide_negative,
        )
        s2_guider = get_value_at_index(cfgguider_17, 0)

        # 9e) KSamplerSelect (Node 20) - "euler"
        ksamplerselect_20 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
        s2_sampler = get_value_at_index(ksamplerselect_20, 0)

        # 9f) BasicScheduler (Node 21) - linear_quadratic, 4 steps, denoise=0.42
        # model from Stage 2 LTXDirectorGuide (node 132, output 3)
        basicscheduler_21 = basicscheduler.EXECUTE_NORMALIZED(
            model=s2_guide_model,
            scheduler="linear_quadratic",
            steps=4,
            denoise=0.42,
        )
        s2_sigmas = get_value_at_index(basicscheduler_21, 0)

        # 9g) SamplerCustomAdvanced (Node 19) - Stage 2 sampling
        # Uses same noise (node 30) as Stage 1
        print("  Running Stage 2 SamplerCustomAdvanced (node 19)...")
        samplercustomadvanced_19 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=shared_noise,
            guider=s2_guider,
            sampler=s2_sampler,
            sigmas=s2_sigmas,
            latent_image=s2_combined_latent,
        )
        s2_output = get_value_at_index(samplercustomadvanced_19, 0)

        print("  Stage 2 sampling complete!")

        # Memory cleanup after Stage 2 sampling
        del cfgguider_17, s2_guider, s2_sigmas, s2_combined_latent
        del upsampled_latent, crop55_latent, crop55_positive, crop55_negative
        del s1_guide_positive, s1_guide_negative, s1_guide_model
        clear_memory()


        # ==================================================================
        # SECTION 10: FINAL DECODE AND OUTPUT
        # ==================================================================
        # Decode the Stage 2 output into video frames and audio,
        # then combine into final MP4 output.
        # Flow: LTXVSeparateAVLatent (22) -> LTXDirectorCropGuides (54)
        #       -> VAEDecode (1) + LTXVAudioVAEDecode (24)
        #       -> VHS_VideoCombine (139)
        # ==================================================================

        print("\n[SECTION 10] Final Decode & Output")
        print("-" * 60)

        # 10a) LTXVSeparateAVLatent (Node 22)
        # Separates Stage 2 output into video and audio latents
        ltxvseparateavlatent_22 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=s2_output
        )
        s2_video_latent = get_value_at_index(ltxvseparateavlatent_22, 0)  # video
        s2_audio_latent = get_value_at_index(ltxvseparateavlatent_22, 1)  # audio

        # 10b) LTXDirectorCropGuides (Node 54)
        # Inputs: positive/negative from Stage 2 LTXDirectorGuide (node 132, outputs 0,1)
        #         latent = video_latent from node 22 (output 0)
        ltxdirectorcropguides_54 = ltxdirectorcropguides.EXECUTE_NORMALIZED(
            positive=s2_guide_positive,
            negative=s2_guide_negative,
            latent=s2_video_latent,
        )
        crop54_latent = get_value_at_index(ltxdirectorcropguides_54, 2)  # output 2: latent for decode

        # 10c) VAEDecode (Node 1)
        # Decodes the cropped latent into image frames
        # Uses video_vae from node 36
        print("  Decoding video latent...")
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_1 = vaedecode.decode(
            samples=crop54_latent,
            vae=video_vae,
        )
        decoded_images = get_value_at_index(vaedecode_1, 0)

        # Memory cleanup after video decode
        del crop54_latent, s2_video_latent, s2_output
        del video_vae, vaeloader_36
        clear_memory()

        # 10d) LTXVAudioVAEDecode (Node 24)
        # Decodes audio latent using audio VAE (node 8)
        print("  Decoding audio latent...")
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        ltxvaudiovaedecode_24 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=s2_audio_latent,
            audio_vae=audio_vae,
        )
        decoded_audio = get_value_at_index(ltxvaudiovaedecode_24, 0)

        # Memory cleanup after audio decode
        del s2_audio_latent, audio_vae, vaeloader_8
        clear_memory()

        # 10e) VHS_VideoCombine (Node 139)
        # Final video output: h264-mp4, yuv420p, crf=8, 24fps
        print("  Combining video and audio into final output...")
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        vhs_videocombine_139 = vhs_videocombine.EXECUTE_NORMALIZED(
            images=decoded_images,
            audio=decoded_audio,
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

        print("\n" + "=" * 60)
        print("VIDEO GENERATION COMPLETE!")
        print(f"Output: LTX2.3/Video - {width}x{height} @ {fps}fps")
        print(f"Duration: {duration_seconds}s ({duration_frames} frames)")
        print("Format: H.264 MP4, YUV420P, CRF=8")
        print("=" * 60)

        # Final memory cleanup
        del decoded_images, decoded_audio
        del director_model, director_positive, director_guide_data
        del director_motion_guide, director_frame_rate
        del s2_guide_positive, s2_guide_negative, s2_guide_model
        clear_memory()

        return vhs_videocombine_139



# ==============================================================================
# SECTION 11: SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    # Modify these parameters to customize the generation.
    # All defaults match the original JSON workflow exactly.

    # --- Generation Parameters ---
    SEED = 0                          # Random seed (node 30, fixed)
    WIDTH = 1280                      # Output width
    HEIGHT = 720                      # Output height
    FPS = 24                          # Frame rate
    DURATION_SECONDS = 31.5           # Video duration in seconds
    DURATION_FRAMES = 756             # Total frames (31.5s * 24fps = 756)

    # --- Image Keyframes (5 segments) ---
    # These are the reference images for each timeline segment.
    # Place them in /content/ComfyUI/input/whatdreamscost/
    IMAGE_PATHS = [
        "whatdreamscost/1.png",       # Segment 1: frames 0 - 226 (9.42s)
        "whatdreamscost/2.png",       # Segment 2: frames 226 - 387 (6.72s)
        "whatdreamscost/3.png",       # Segment 3: frames 387 - 519 (5.48s)
        "whatdreamscost/4.png",       # Segment 4: frames 519 - 744 (9.40s)
        "whatdreamscost/5.3.png",     # Segment 5: frames 744 - 756 (0.49s)
    ]

    # --- Audio Configuration ---
    # The audio file should be placed in /content/ComfyUI/input/whatdreamscost/
    AUDIO_FILE = "whatdreamscost/Late night trap.mp3"

    # --- Global Prompt ---
    # Uses the full cinematic prompt defined above (GLOBAL_PROMPT)
    # You can override it here if needed
    PROMPT = GLOBAL_PROMPT

    # ==========================================================================
    # EXECUTION
    # ==========================================================================

    print("\n" + "=" * 60)
    print("LTX 2.3 DIRECTOR 2.0 - MUSIC VIDEO PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Seed: {SEED}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  FPS: {FPS}")
    print(f"  Duration: {DURATION_SECONDS}s ({DURATION_FRAMES} frames)")
    print(f"  Images: {len(IMAGE_PATHS)} keyframe segments")
    print(f"  Audio: {AUDIO_FILE}")
    print(f"\nStage 1: 8 steps, denoise=1.0, guide_weight=0.5")
    print(f"Stage 2: 4 steps, denoise=0.42, guide_weight=1.0")
    print()

    # --- Step 1: Environment Setup ---
    # Uncomment these when running in Google Colab for the first time:
    # install_packages()
    # clone_repos()
    # install_system_packages()

    # --- Step 2: Download Models ---
    # Uncomment when running for the first time:
    # download_all_models()

    # --- Step 3: Generate Video ---
    result = main_ltx_director_mv(
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        duration_seconds=DURATION_SECONDS,
        duration_frames=DURATION_FRAMES,
        global_prompt=PROMPT,
        image_paths=IMAGE_PATHS,
        audio_file_path=AUDIO_FILE,
    )

    # --- Step 4: Display Video (in Colab) ---
    # Uncomment to display the generated video in notebook:
    # output_dir = "/content/ComfyUI/output/LTX2.3"
    # import glob
    # videos = sorted(glob.glob(os.path.join(output_dir, "Video_*.mp4")))
    # if videos:
    #     display_video(videos[-1])

    # --- Final Memory Cleanup ---
    clear_memory()
    print("\nPipeline finished. Memory cleaned up.")
