"""Z-Image Predictor for Replicate with detailed logging and error handling."""
import sys
import time
import traceback
from pathlib import Path as FilePath
from typing import Union

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

# Force stdout/stderr to be unbuffered for real-time logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
MODEL_CACHE = "./model-cache"


def log(message: str):
    """Print timestamped log message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def download_model():
    """Download model weights if not exists."""
    if not FilePath(MODEL_CACHE).exists():
        log(f"Downloading model {MODEL_NAME}...")
        snapshot_download(
            MODEL_NAME,
            local_dir=MODEL_CACHE,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        log("Model download completed!")
    else:
        log(f"Model already exists at {MODEL_CACHE}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Z-Image model into memory."""
        try:
            log("=" * 60)
            log("Starting Z-Image-Turbo setup...")
            log("=" * 60)
            
            # Check CUDA availability
            log(f"PyTorch version: {torch.__version__}")
            log(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                log(f"CUDA version: {torch.version.cuda}")
                log(f"GPU: {torch.cuda.get_device_name(0)}")
                log(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Download model if needed
            log("Checking model weights...")
            download_model()
            
            log(f"Model cache found at: {MODEL_CACHE}")
            
            # Load pipeline
            log("Loading Z-Image-Turbo pipeline...")
            start_time = time.time()
            
            self.pipe = DiffusionPipeline.from_pretrained(
                MODEL_CACHE,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
            
            log(f"Pipeline loaded in {time.time() - start_time:.2f}s")
            
            # Move to GPU
            log("Moving model to CUDA...")
            start_time = time.time()
            self.pipe.to("cuda")
            log(f"Model moved to CUDA in {time.time() - start_time:.2f}s")
            
            # Enable optimizations
            log("Enabling optimizations...")
            
            # Try to enable Flash Attention
            try:
                if hasattr(self.pipe, 'transformer'):
                    self.pipe.transformer.set_attention_backend("flash")
                    log("✓ Flash Attention enabled")
            except Exception as e:
                log(f"⚠ Flash Attention not available: {e}")
            
            # Enable memory efficient attention
            try:
                self.pipe.enable_attention_slicing()
                log("✓ Attention slicing enabled")
            except Exception as e:
                log(f"⚠ Attention slicing failed: {e}")
            
            # Warmup run
            log("Running warmup inference...")
            start_time = time.time()
            _ = self.pipe(
                prompt="test",
                height=512,
                width=512,
                num_inference_steps=1,
                guidance_scale=0.0,
            )
            log(f"Warmup completed in {time.time() - start_time:.2f}s")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                log(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
            log("=" * 60)
            log("✓ Setup completed successfully!")
            log("=" * 60)
            
        except Exception as e:
            log("=" * 60)
            log(f"✗ SETUP FAILED: {str(e)}")
            log("=" * 60)
            log("Full traceback:")
            traceback.print_exc()
            raise

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation. Supports English and Chinese.",
            default="A beautiful landscape with mountains and a lake at sunset"
        ),
        width: int = Input(
            description="Width of the output image",
            default=1024,
            ge=512,
            le=2048
        ),
        height: int = Input(
            description="Height of the output image", 
            default=1024,
            ge=512,
            le=2048
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (Z-Image-Turbo uses 8 steps)",
            default=8,
            ge=1,
            le=50
        ),
        seed: Union[int, None] = Input(
            description="Random seed for reproducibility. Leave blank for random generation.",
            default=None
        ),
    ) -> Path:
        """Generate an image from a text prompt using Z-Image-Turbo."""
        
        try:
            log("=" * 60)
            log("Starting prediction...")
            log(f"Prompt: {prompt}")
            log(f"Size: {width}x{height}")
            log(f"Steps: {num_inference_steps}")
            log(f"Seed: {seed}")
            log("=" * 60)
            
            # Set random seed if provided
            generator = None
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)
                log(f"Using seed: {seed}")
            
            # Generate image
            log("Generating image...")
            start_time = time.time()
            
            image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,  # Z-Image-Turbo uses guidance_scale=0
                generator=generator,
            ).images[0]
            
            generation_time = time.time() - start_time
            log(f"✓ Image generated in {generation_time:.2f}s")
            
            # Save output
            output_path = "/tmp/output.png"
            log(f"Saving image to {output_path}...")
            image.save(output_path, format="PNG", optimize=True)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                log(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            log("=" * 60)
            log("✓ Prediction completed successfully!")
            log("=" * 60)
            
            return Path(output_path)
            
        except Exception as e:
            log("=" * 60)
            log(f"✗ PREDICTION FAILED: {str(e)}")
            log("=" * 60)
            log("Full traceback:")
            traceback.print_exc()
            raise
