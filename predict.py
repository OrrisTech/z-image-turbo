"""Z-Image Predictor for Replicate."""
from cog import BasePredictor, Input, Path
import torch
from diffusers import ZImagePipeline
from typing import Optional

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the Z-Image model into memory."""
        print("Loading Z-Image-Turbo model...")
        
        self.pipe = ZImagePipeline.from_pretrained(
            "./model-cache",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to("cuda")
        
        # Enable Flash Attention for better performance
        try:
            self.pipe.transformer.set_attention_backend("flash")
            print("Flash Attention enabled")
        except Exception as e:
            print(f"Flash Attention not available: {e}")
        
        print("Model loaded successfully!")

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
            default=9,
            ge=1,
            le=50
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility. Leave blank for random generation.",
            default=None
        ),
    ) -> Path:
        """Generate an image from a text prompt using Z-Image-Turbo."""
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        print(f"Generating image with prompt: {prompt}")
        print(f"Size: {width}x{height}, Steps: {num_inference_steps}, Seed: {seed}")
        
        # Generate image
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # Z-Image-Turbo uses guidance_scale=0
            generator=generator,
        ).images[0]
        
        # Save output
        output_path = "/tmp/output.png"
        image.save(output_path)
        
        return Path(output_path)
