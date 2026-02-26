"""Download Z-Image model weights from Hugging Face with progress logging."""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
MODEL_CACHE = "./model-cache"


def log(message: str):
    """Print timestamped log message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def download_weights():
    """Download model weights to local cache."""
    try:
        log("=" * 60)
        log(f"Starting download: {MODEL_NAME}")
        log(f"Target directory: {MODEL_CACHE}")
        log("=" * 60)
        
        # Check if already downloaded
        if Path(MODEL_CACHE).exists():
            log(f"⚠ Model cache already exists at {MODEL_CACHE}")
            log("Checking for updates...")
        
        start_time = time.time()
        
        # Download with progress
        log("Downloading from Hugging Face...")
        snapshot_download(
            MODEL_NAME,
            local_dir=MODEL_CACHE,
            local_dir_use_symlinks=False,
            resume_download=True,  # Resume if interrupted
        )
        
        download_time = time.time() - start_time
        log(f"✓ Download completed in {download_time:.2f}s")
        
        # Verify download
        model_path = Path(MODEL_CACHE)
        if not model_path.exists():
            raise FileNotFoundError(f"Model cache not found after download: {MODEL_CACHE}")
        
        # List downloaded files
        files = list(model_path.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        log(f"Total files: {len([f for f in files if f.is_file()])}")
        log(f"Total size: {total_size / 1024**3:.2f} GB")
        
        log("=" * 60)
        log("✓ Model weights ready!")
        log("=" * 60)
        
    except Exception as e:
        log("=" * 60)
        log(f"✗ DOWNLOAD FAILED: {str(e)}")
        log("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    download_weights()
