# Z-Image Replicate Deployment

Deploy [Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image) to Replicate.

**🚀 Replicate Model**: [`r8.im/leizeng/z-image-turbo`](https://replicate.com/leizeng/z-image-turbo)

## Model Information

- **Model**: Z-Image-Turbo (Tongyi-MAI)
- **Parameters**: 6B
- **Inference Steps**: 8 NFEs
- **VRAM**: 16GB
- **Features**: 
  - Sub-second inference on H800 GPUs
  - Bilingual text rendering (English & Chinese)
  - Photorealistic image generation
  - Strong instruction adherence

## Project Structure

```
zimage-replicate-model/
├── cog.yaml                # Cog configuration
├── predict.py              # Prediction interface (with detailed logging)
├── requirements.txt        # Python dependencies
├── download_weights.py     # Model weight downloader
├── TROUBLESHOOTING.md      # Debugging guide
└── README.md              # This file
```

## Key Features

- ✅ **Detailed Logging**: Timestamped logs for every operation
- ✅ **Error Handling**: Full tracebacks on failures
- ✅ **Auto-download Model**: Model downloaded on first run (~12GB)
- ✅ **Optimized Setup**: Warmup run for faster predictions
- ✅ **Resource Monitoring**: GPU memory usage tracking
- ✅ **Unbuffered Output**: Real-time log visibility

## Local Development

### Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- Docker installed and running
- NVIDIA GPU with CUDA support (for testing)

### Build the model

```bash
cog build
```

This will:
1. Create a Docker container with CUDA 12.1
2. Install Python dependencies
3. Download Z-Image-Turbo weights from Hugging Face

### Test locally

```bash
cog predict -i prompt="A beautiful Chinese landscape painting"
```

### Test with custom parameters

```bash
cog predict \
  -i prompt="Young woman in traditional Hanfu dress" \
  -i width=1024 \
  -i height=1024 \
  -i num_inference_steps=9 \
  -i seed=42
```

## Deploy to Replicate

### 1. Push to Replicate

```bash
cog login
cog push r8.im/leizeng/z-image-turbo
```

### 2. Use the API

```python
import replicate

output = replicate.run(
    "leizeng/z-image-turbo:latest",
    input={
        "prompt": "A serene mountain landscape at sunset",
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
)
print(output)
```

## Parameters

- **prompt** (string): Text description for image generation. Supports English and Chinese.
- **width** (integer, 512-2048): Output image width. Default: 1024
- **height** (integer, 512-2048): Output image height. Default: 1024
- **num_inference_steps** (integer, 1-50): Number of denoising steps. Default: 9 (8 actual steps)
- **seed** (integer, optional): Random seed for reproducibility

## Performance

- **Build time**: 10-15 minutes (downloading 12GB model)
- **Setup time**: 30-60 seconds (loading model to GPU)
- **Inference time**: 2-5 seconds (1024x1024, 8 steps)
- **VRAM usage**: ~16GB
- **Recommended resolution**: 1024x1024

## Troubleshooting

If you encounter issues (model stuck, no logs, etc.), see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging guide.

## License

Z-Image is released under Apache 2.0 license. See [LICENSE](https://github.com/Tongyi-MAI/Z-Image/blob/main/LICENSE).

## References

- [Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Cog Documentation](https://cog.run)
- [Replicate Documentation](https://replicate.com/docs)
