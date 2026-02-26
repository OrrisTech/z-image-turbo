"""Download Z-Image model weights from Hugging Face."""
import os
from huggingface_hub import snapshot_download

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
MODEL_CACHE = "./model-cache"

def download_weights():
    """Download model weights to local cache."""
    print(f"Downloading {MODEL_NAME}...")
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_CACHE,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {MODEL_CACHE}")

if __name__ == "__main__":
    download_weights()
