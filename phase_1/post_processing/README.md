# Cookbook Image Post-Processing

This directory contains tools and scripts for correcting distortions in scanned cookbook page images.

## Overview

The processing pipeline corrects:
- Curved pages from book binding
- Perspective distortion
- Non-uniform warping
- Text line curvature

## Environment Setup

### Prerequisites
- Conda or Miniconda installed
- Python 3.11 or higher

### Create Conda Environment

**Bash/Linux/macOS:**
```bash
# Create new conda environment
conda create -n cookbook-processing python=3.11 -y

# Activate the environment
conda activate cookbook-processing
```

**PowerShell/Windows:**
```powershell
# Create new conda environment
conda create -n cookbook-processing python=3.11 -y

# Activate the environment
conda activate cookbook-processing
```

### Install Dependencies

**Bash/Linux/macOS:**
```bash
# Install core dependencies
conda install -c conda-forge opencv numpy matplotlib pillow scipy -y

# Alternative: Install via pip
pip install opencv-python numpy matplotlib pillow scipy scikit-image imutils
```

**PowerShell/Windows:**
```powershell
# Install core dependencies
conda install -c conda-forge opencv numpy matplotlib pillow scipy -y

# Alternative: Install via pip
pip install opencv-python numpy matplotlib pillow scipy scikit-image imutils
```

### Verify Installation

**Bash/Linux/macOS:**
```bash
python -c "import cv2; import numpy as np; print('OpenCV version:', cv2.__version__); print('NumPy version:', np.__version__)"
```

**PowerShell/Windows:**
```powershell
python -c "import cv2; import numpy as np; print('OpenCV version:', cv2.__version__); print('NumPy version:', np.__version__)"
```

Expected output:
```
OpenCV version: 4.x.x
NumPy version: 1.x.x
```

## Project Structure

```
post_processing/
├── README.md                    # This file
├── IMAGE_PROCESSING_PLAN.md     # Detailed project plan
├── scripts/                     # Processing scripts (to be created)
│   ├── dewarp.py               # Main processing script
│   ├── test_samples.py         # Testing script
│   └── utils.py                # Helper functions
├── test_outputs/               # Test results during development
├── processed_images/           # Final corrected images
└── processing_log.txt          # Processing report
```

## Quick Start

1. **Setup environment** (see above)
2. **Activate environment**: `conda activate cookbook-processing`
3. **Run test script**: `python scripts/test_samples.py`
4. **Process all images**: `python scripts/dewarp.py`

## Dependencies Reference

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.5 | Image processing, dewarping |
| numpy | ≥1.20 | Array operations |
| matplotlib | ≥3.3 | Visualization |
| pillow | ≥8.0 | Image I/O |
| scipy | ≥1.6 | Scientific computing |
| scikit-image | ≥0.18 | Additional image processing |
| imutils | ≥0.5 | OpenCV convenience functions |

## Troubleshooting

### OpenCV Import Error (Linux)
If you get `ImportError: libGL.so.1: cannot open shared object file`:
```bash
# Linux
conda install -c conda-forge libgl

# Or use headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

### Conda Environment Issues

**Bash/Linux/macOS:**
```bash
# Remove and recreate environment
conda deactivate
conda env remove -n cookbook-processing
# Then recreate using steps above
```

**PowerShell/Windows:**
```powershell
# Remove and recreate environment
conda deactivate
conda env remove -n cookbook-processing
# Then recreate using steps above
```

## Next Steps

See `IMAGE_PROCESSING_PLAN.md` for the complete processing workflow and implementation details.
