import subprocess
from pathlib import Path

MODELS_DIR = Path("data/models")


def convert_model(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
    batch_size: int = 32
):
    """Convert ONNX to TensorRT engine using trtexec."""
    
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=input:1x3x112x112",
        f"--optShapes=input:{batch_size}x3x112x112",
        f"--maxShapes=input:64x3x112x112",
    ]
    
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", f"--calib={MODELS_DIR}/calibration_cache.txt"])
    
    print(f"🔧 Converting {onnx_path.name} to {precision.upper()}...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Conversion successful! Engine saved to {engine_path}")
    else:
        print(f"❌ Conversion failed!")
        print(result.stderr)


def main():
    onnx_model = MODELS_DIR / "arcface.onnx"
    
    if not onnx_model.exists():
        print(f"❌ ONNX model not found: {onnx_model}")
        print("Run: poetry run python scripts/download_models.py")
        return
    
    # Convert to different precisions
    for precision in ["fp32", "fp16"]:
        engine_path = MODELS_DIR / f"arcface_{precision}.trt"
        convert_model(onnx_model, engine_path, precision)
    
    print()
    print("✅ All conversions complete!")
    print()
    print("Available engines:")
    for engine in MODELS_DIR.glob("*.trt"):
        size_mb = engine.stat().st_size / (1024 * 1024)
        print(f"  - {engine.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()