#!/usr/bin/env python3
"""
Quick test to verify all dependencies for app.py
"""

import sys

print("="*80)
print("CHECKING DEPENDENCIES FOR app.py")
print("="*80)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check required packages
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'transformers': 'transformers',
    'sklearn': 'scikit-learn',
    'tqdm': 'tqdm',
    'pyarrow': 'pyarrow'
}

missing_packages = []
print("\nChecking installed packages:")
for import_name, package_name in required_packages.items():
    try:
        if import_name == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(import_name)
            version = mod.__version__
        print(f"  ✓ {package_name:20s} {version}")
    except ImportError:
        print(f"  ✗ {package_name:20s} NOT INSTALLED")
        missing_packages.append(package_name)

# Check for embed_texts module
print("\nChecking required modules:")
try:
    from embed_texts import TextExtractor, AraBERTEmbedder
    print("  ✓ embed_texts.py (TextExtractor, AraBERTEmbedder)")
except ImportError as e:
    print(f"  ✗ embed_texts.py - ERROR: {e}")
    missing_packages.append("embed_texts.py")

# Check for metadata file
print("\nChecking required files:")
from pathlib import Path

metadata_file = Path("openiti_metadata.parquet")
if metadata_file.exists():
    size_mb = metadata_file.stat().st_size / (1024*1024)
    print(f"  ✓ openiti_metadata.parquet ({size_mb:.2f} MB)")
else:
    print(f"  ✗ openiti_metadata.parquet NOT FOUND")
    missing_packages.append("openiti_metadata.parquet")

db_dir = Path("../db")
if db_dir.exists() and db_dir.is_dir():
    text_count = sum(1 for _ in db_dir.rglob("*") if _.is_file() and not _.name.startswith('.'))
    print(f"  ✓ ../db/ directory ({text_count:,} files)")
else:
    print(f"  ✗ ../db/ directory NOT FOUND")
    missing_packages.append("db/ directory")

# Check device (MPS for M1)
print("\nChecking compute device:")
try:
    import torch
    if torch.backends.mps.is_available():
        print("  ✓ MPS (Metal Performance Shaders) available - M1/M2/M3 Mac acceleration")
    elif torch.cuda.is_available():
        print("  ✓ CUDA available - GPU acceleration")
    else:
        print("  ⚠ CPU only - slower but will work")
except:
    print("  ? Could not check device")

# Summary
print("\n" + "="*80)
if missing_packages:
    print("❌ MISSING DEPENDENCIES:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\nPlease install missing packages before running app.py")
    print("\nTo install Python packages:")
    print("  pip install <package_name>")
else:
    print("✅ ALL DEPENDENCIES SATISFIED")
    print("\nYou can run the analyzer with:")
    print("  python app.py")
print("="*80)

