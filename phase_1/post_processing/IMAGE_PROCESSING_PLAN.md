# Cookbook Image Processing Plan

## Project Goal
Process 224 cookbook page images to correct distortions from curved/folded pages, making them suitable for OCR and digital use.

## Current Status
- **Phase**: Planning
- **Images**: 224 JPG files in `phase_1/cookbook_images/`
- **Environment**: Conda (to be created)

---

## Approach Strategy

### Stage 1: Environment Setup
- [ ] Create conda environment
- [ ] Install dependencies (OpenCV, NumPy, etc.)
- [ ] Test installation with sample image

### Stage 2: Analysis & Testing
- [ ] Load and inspect sample images (pages 1, 50, 100, 150, 200)
- [ ] Identify common distortion patterns
- [ ] Test multiple dewarping approaches on samples
- [ ] Compare results and select best method

### Stage 3: Pipeline Development
- [ ] Implement page detection (edge/corner detection)
- [ ] Implement perspective correction
- [ ] Implement dewarping algorithm
- [ ] Add quality checks and validation
- [ ] Create batch processing script

### Stage 4: Batch Processing
- [ ] Process all 224 images
- [ ] Save corrected images to output directory
- [ ] Generate processing report/log
- [ ] Visual QA on sample outputs

---

## Technical Approach Options

### Option A: OpenCV Custom Pipeline (Recommended for Control)
**Pros**: Full control, no external dependencies, well-documented
**Cons**: More manual implementation
**Steps**:
1. Grayscale conversion & preprocessing
2. Edge detection (Canny)
3. Contour detection for page boundaries
4. Perspective transform (4-point)
5. Cylindrical/polynomial dewarping for curves
6. Post-processing (contrast, sharpening)

### Option B: page-dewarp Library
**Pros**: Specialized for book scanning, handles 3D curvature
**Cons**: May need tuning, less maintained
**Steps**:
1. Install page-dewarp
2. Configure parameters
3. Apply to images
4. Post-process as needed

### Option C: Hybrid Approach (Selected)
**Combine OpenCV + specialized dewarping**
1. OpenCV for page detection & perspective
2. Custom or library-based dewarping for curves
3. OpenCV for final cleanup

---

## Dependencies to Install

```bash
# Core libraries
- opencv-python
- numpy
- matplotlib (for visualization)
- pillow
- scipy

# Optional/Testing
- scikit-image
- imutils
```

---

## Output Structure

```
phase_1/
├── cookbook_images/          # Original images (224 files)
├── processed_images/         # Corrected images (to be created)
├── test_outputs/            # Test results during development
├── processing_log.txt       # Processing report
└── scripts/                 # Processing scripts
    ├── dewarp.py           # Main processing script
    ├── test_samples.py     # Testing script
    └── utils.py            # Helper functions
```

---

## Success Criteria
- [ ] All 224 images processed without errors
- [ ] Text lines are straightened (no visible curve)
- [ ] Page boundaries are properly aligned
- [ ] No significant quality loss
- [ ] Images ready for OCR processing

---

## Notes & Observations
*To be filled during processing*

---

## Timeline
- **Setup**: ~15 minutes
- **Testing**: ~30-45 minutes
- **Development**: ~1-2 hours
- **Batch Processing**: ~10-30 minutes (depending on method)
- **QA**: ~15 minutes

**Total Estimated Time**: 2-3 hours
