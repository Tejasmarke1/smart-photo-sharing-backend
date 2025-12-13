import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "retinaface": {
        "url": "https://huggingface.co/felixrosberg/RetinaFace/resolve/main/RetinaFace-Res50.h5?download=true",
        "path": MODELS_DIR / "retinaface.h5",
        "size": "1.6 GB"
    },
    "arcface_onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "path": MODELS_DIR / "arcface.onnx",
        "size": "143 MB"
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(name: str, config: dict):
    if config["path"].exists():
        print(f"✅ {name} already exists, skipping...")
        return
    
    print(f"⬇️  Downloading {name} ({config['size']})...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=name) as t:
        urllib.request.urlretrieve(
            config["url"],
            filename=config["path"],
            reporthook=t.update_to
        )
    
    print(f"✅ {name} downloaded successfully!")


def main():
    print("🤖 Downloading face recognition models...")
    print(f"Models will be saved to: {MODELS_DIR}")
    print()
    
    for name, config in MODELS.items():
        try:
            download_model(name, config)
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
    
    print()
    print("✅ All models downloaded!")
    print()
    print("⚠️  Note: TensorRT engines must be built locally:")
    print("    poetry run python scripts/convert_to_tensorrt.py")


if __name__ == "__main__":
    main()