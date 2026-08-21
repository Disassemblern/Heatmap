"""Report GPU availability and the versions of the pipeline's dependencies."""

from __future__ import annotations

from importlib import metadata

PACKAGES = (
    "torch",
    "ultralytics",
    "supervision",
    "deep-sort-realtime",
    "opencv-python",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "mplsoccer",
    "matplotlib",
    "pyyaml",
)


def main() -> None:
    import torch

    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    print(f"CUDA version:   {torch.version.cuda}")
    if available:
        properties = torch.cuda.get_device_properties(0)
        vram_gb = properties.total_memory / 1e9
        print(f"GPU:            {properties.name} ({vram_gb:.1f} GB)")

    print("\nPackages:")
    for package in PACKAGES:
        try:
            print(f"  {package:20s} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"  {package:20s} NOT INSTALLED")


if __name__ == "__main__":
    main()
