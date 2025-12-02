# Design Document

## Overview

The cookbook dewarp system is an image processing pipeline built with OpenCV and Python that corrects distortions in scanned cookbook page images. The system uses a hybrid approach combining traditional computer vision techniques for page detection and perspective correction with specialized dewarping algorithms for curved page surfaces.

The pipeline processes 224 JPG images through five main stages:
1. **Spine Detection & Splitting**: Identifies book spine in two-page spreads and splits into individual pages
2. **Preprocessing**: Image loading, grayscale conversion, and noise reduction
3. **Page Detection**: Edge detection and contour analysis to find page boundaries
4. **Perspective Correction**: Four-point transformation to align pages rectangularly
5. **Cylindrical Dewarping**: Curve correction for text line straightening

The system handles both two-page spreads (most images) and single-page covers (first and last images), supports test mode (for algorithm validation on samples) and batch mode (for processing all images), with comprehensive logging and error handling.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Processing Script                   │
│                        (dewarp.py)                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├──► Spine Detection & Splitting Module
                 │    - Two-page spread detection
                 │    - Spine location identification
                 │    - Spine angle detection
                 │    - Image splitting into left/right pages
                 │
                 ├──► Preprocessing Module
                 │    - Image loading
                 │    - Grayscale conversion
                 │    - Noise reduction
                 │    - Contrast enhancement
                 │
                 ├──► Page Detection Module
                 │    - Edge detection (Canny)
                 │    - Contour finding
                 │    - Corner point extraction
                 │    - Boundary validation
                 │
                 ├──► Perspective Correction Module
                 │    - Four-point transform calculation
                 │    - Aspect ratio preservation
                 │    - Warp application
                 │
                 ├──► Dewarping Module
                 │    - Text line detection
                 │    - Curvature analysis
                 │    - Cylindrical unwarp
                 │    - Polynomial correction
                 │
                 ├──► Post-processing Module
                 │    - Sharpening
                 │    - Contrast adjustment
                 │    - Quality validation
                 │
                 └──► Logging & I/O Module
                      - File operations
                      - Progress tracking
                      - Error logging
                      - Statistics collection
```

### Processing Flow

```
Source Image (JPG)
    │
    ├──► Load & Validate
    │
    ├──► Detect Spread Type (Single vs Two-Page)
    │
    ├──► IF Two-Page Spread:
    │    ├──► Detect Spine Location
    │    ├──► Detect Spine Angle
    │    └──► Split into Left & Right Page Images
    │
    ├──► FOR EACH Page Image:
    │    │
    │    ├──► Preprocess
    │    │    └──► Grayscale, Denoise, Enhance
    │    │
    │    ├──► Detect Page Boundary
    │    │    ├──► Edge Detection
    │    │    ├──► Find Contours
    │    │    └──► Extract Corners
    │    │
    │    ├──► Apply Perspective Transform
    │    │    └──► 4-point Warp
    │    │
    │    ├──► Apply Cylindrical Dewarping
    │    │    ├──► Detect Text Lines
    │    │    ├──► Measure Curvature
    │    │    └──► Apply Correction
    │    │
    │    ├──► Post-process
    │    │    └──► Sharpen, Adjust
    │    │
    │    └──► Save Corrected Image
    │         └──► Log Results
```

## Components and Interfaces

### 1. Main Processing Script (`dewarp.py`)

**Purpose**: Orchestrates the entire processing pipeline and handles batch operations.

**Key Functions**:
- `main()`: Entry point, handles command-line arguments
- `process_batch(input_dir, output_dir)`: Processes all images in directory
- `process_single_image(image_path, output_path)`: Processes one image through pipeline
- `setup_directories()`: Creates output directories if needed

**Interface**:
```python
def process_single_image(image_path: str, output_path: str) -> dict:
    """
    Process a single image through the dewarp pipeline.
    
    Args:
        image_path: Path to source image
        output_path: Path to save corrected image
        
    Returns:
        dict: Processing result with keys:
            - success: bool
            - error: str (if failed)
            - processing_time: float
            - dimensions: tuple (width, height)
    """
```

### 2. Spine Detection & Splitting Module (`utils.py`)

**Purpose**: Identifies book spine in two-page spreads and splits images into individual pages.

**Key Functions**:
- `detect_spread_type(image)`: Determines if image is single-page or two-page spread
- `detect_spine_location(image)`: Finds the vertical line representing the book spine
- `detect_spine_angle(image, spine_x)`: Measures spine orientation (handles non-vertical spines)
- `split_along_spine(image, spine_x, spine_angle)`: Splits image into left and right pages

**Interface**:
```python
def detect_spine_location(image: np.ndarray) -> Optional[Tuple[int, float]]:
    """
    Detect the book spine location and angle in a two-page spread.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Tuple of (spine_x_position, spine_angle_degrees) or None if not detected
        spine_x_position: X-coordinate of spine center
        spine_angle_degrees: Angle of spine from vertical (0 = perfectly vertical)
    """

def split_along_spine(image: np.ndarray, spine_x: int, spine_angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a two-page spread into left and right page images.
    
    Args:
        image: Source image containing two-page spread
        spine_x: X-coordinate of spine center
        spine_angle: Angle of spine from vertical in degrees
        
    Returns:
        Tuple of (left_page, right_page) as numpy arrays
    """
```

**Spine Detection Approach**:
1. **Vertical Line Detection**: Use Hough Line Transform to find strong vertical lines near image center
2. **Shadow/Darkness Detection**: Detect dark region in center (spine shadow) using intensity analysis
3. **Symmetry Analysis**: Analyze left/right symmetry to confirm spine location
4. **Angle Estimation**: Measure spine angle to handle non-vertical orientations

### 3. Preprocessing Module (`utils.py`)

**Purpose**: Prepares images for analysis and correction.

**Key Functions**:
- `load_image(path)`: Loads image from disk
- `preprocess(image)`: Applies grayscale conversion and noise reduction
- `enhance_contrast(image)`: Improves edge detection quality

**Interface**:
```python
def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for page detection.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        np.ndarray: Preprocessed grayscale image
    """
```

### 4. Page Detection Module (`utils.py`)

**Purpose**: Identifies page boundaries within scanned images.

**Key Functions**:
- `detect_page_boundary(image)`: Finds page corners
- `find_largest_contour(contours)`: Selects page contour
- `order_points(corners)`: Orders corners consistently (TL, TR, BR, BL)
- `validate_quadrilateral(corners)`: Checks if corners form valid page shape

**Interface**:
```python
def detect_page_boundary(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the four corner points of the page boundary.
    
    Args:
        image: Preprocessed grayscale image
        
    Returns:
        np.ndarray: 4x2 array of corner coordinates [TL, TR, BR, BL]
                   or None if detection fails
    """
```

### 5. Perspective Correction Module (`utils.py`)

**Purpose**: Applies geometric transformation to align pages rectangularly.

**Key Functions**:
- `calculate_perspective_transform(corners, target_size)`: Computes transform matrix
- `apply_perspective_warp(image, corners)`: Applies transformation
- `compute_target_dimensions(corners)`: Calculates output size preserving aspect ratio

**Interface**:
```python
def apply_perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to correct viewing angle.
    
    Args:
        image: Source image
        corners: 4x2 array of page corners
        
    Returns:
        np.ndarray: Perspective-corrected image
    """
```

### 6. Dewarping Module (`utils.py`)

**Purpose**: Corrects curved page surfaces and straightens text lines.

**Key Functions**:
- `detect_text_lines(image)`: Identifies horizontal text regions
- `measure_curvature(text_lines)`: Analyzes curve patterns
- `apply_cylindrical_dewarp(image, curvature_params)`: Applies curve correction
- `create_dewarp_mesh(image_shape, curvature)`: Generates correction grid

**Interface**:
```python
def apply_cylindrical_dewarp(image: np.ndarray) -> np.ndarray:
    """
    Apply cylindrical dewarping to straighten curved pages.
    
    Args:
        image: Perspective-corrected image
        
    Returns:
        np.ndarray: Dewarped image with straightened text lines
    """
```

### 7. Post-processing Module (`utils.py`)

**Purpose**: Enhances final image quality.

**Key Functions**:
- `sharpen_image(image)`: Applies sharpening filter
- `adjust_contrast(image)`: Optimizes contrast for OCR
- `validate_quality(image)`: Checks output meets quality thresholds

**Interface**:
```python
def postprocess(image: np.ndarray) -> np.ndarray:
    """
    Apply final quality enhancements.
    
    Args:
        image: Dewarped image
        
    Returns:
        np.ndarray: Final corrected image
    """
```

### 8. Logging Module (`utils.py`)

**Purpose**: Tracks processing results and errors.

**Key Functions**:
- `init_logger(log_path)`: Sets up logging configuration
- `log_processing_result(filename, result)`: Records individual image result
- `generate_summary_report(results)`: Creates final statistics report

**Interface**:
```python
def log_processing_result(filename: str, result: dict, logger: logging.Logger) -> None:
    """
    Log the processing result for a single image.
    
    Args:
        filename: Name of processed image
        result: Processing result dictionary
        logger: Logger instance
    """
```

### 9. Test Script (`test_samples.py`)

**Purpose**: Validates algorithm on sample images before batch processing.

**Key Functions**:
- `test_on_samples(sample_indices)`: Processes specified test images
- `visualize_comparison(original, corrected)`: Displays before/after
- `save_test_outputs(results)`: Saves test results to test_outputs directory

**Interface**:
```python
def test_on_samples(sample_indices: List[int], input_dir: str, output_dir: str) -> None:
    """
    Process test sample images and display results.
    
    Args:
        sample_indices: List of page numbers to test (e.g., [1, 50, 100])
        input_dir: Source image directory
        output_dir: Test output directory
    """
```

## Data Models

### SpineInfo

Represents detected book spine information.

```python
@dataclass
class SpineInfo:
    detected: bool           # Whether spine was detected
    x_position: int         # X-coordinate of spine center
    angle: float            # Angle from vertical in degrees
    confidence: float       # Detection confidence score (0-1)
    is_two_page: bool      # True if two-page spread, False if single page
```

### ProcessingResult

Represents the outcome of processing a single image.

```python
@dataclass
class ProcessingResult:
    filename: str
    success: bool
    error: Optional[str]
    processing_time: float
    input_dimensions: Tuple[int, int]
    output_dimensions: Optional[Tuple[int, int]]
    is_two_page: bool
    spine_detected: bool
    pages_generated: int  # 1 for single page, 2 for two-page spread
    page_detected: bool
    perspective_applied: bool
    dewarp_applied: bool
```

### PageBoundary

Represents detected page corners.

```python
@dataclass
class PageBoundary:
    corners: np.ndarray  # 4x2 array: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float    # Detection confidence score
    area: float         # Page area in pixels
    
    def is_valid(self) -> bool:
        """Check if boundary forms a valid quadrilateral."""
        pass
```

### CurvatureParams

Represents page curvature characteristics.

```python
@dataclass
class CurvatureParams:
    curve_type: str  # 'cylindrical', 'polynomial', 'none'
    coefficients: np.ndarray  # Polynomial coefficients or curve parameters
    strength: float  # Curvature magnitude (0-1)
    direction: str  # 'horizontal', 'vertical', 'both'
```

### ProcessingConfig

Configuration parameters for the pipeline.

```python
@dataclass
class ProcessingConfig:
    # Edge detection
    canny_low: int = 50
    canny_high: int = 150
    
    # Contour detection
    min_area_ratio: float = 0.5  # Minimum page area as ratio of image
    
    # Dewarping
    dewarp_strength: float = 1.0
    text_line_threshold: int = 100
    
    # Output
    output_quality: int = 95  # JPEG quality
    preserve_color: bool = True
    
    # Processing
    show_progress: bool = True
    save_intermediate: bool = False
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Batch processing completeness
*For any* directory containing N valid JPG files, batch processing should process exactly N files and generate log entries for exactly N files.
**Validates: Requirements 1.1, 7.1**

### Property 2: Two-page spread splitting
*For any* image detected as a two-page spread, the system should generate exactly 2 output page images.
**Validates: Requirements 2.3**

### Property 3: Single-page preservation
*For any* image detected as a single page, the system should generate exactly 1 output page image.
**Validates: Requirements 2.5**

### Property 4: Spine detection consistency
*For any* two-page spread image, if spine detection succeeds, the spine x-position should be within the middle 40% of the image width.
**Validates: Requirements 2.2**

### Property 5: Filename preservation with page suffix
*For any* input image with filename F, the output images should have filenames derived from F with appropriate page suffixes (e.g., F_left, F_right, or F_single).
**Validates: Requirements 1.2**

### Property 6: Error resilience
*For any* batch containing both valid and invalid images, processing should continue after failures, process all valid images, and log all errors.
**Validates: Requirements 1.4**

### Property 7: Page boundary structure
*For any* page image where page detection succeeds, the detected boundary should consist of exactly 4 corner points forming a valid quadrilateral.
**Validates: Requirements 3.1, 3.3**

### Property 8: Largest contour selection
*For any* page image containing multiple quadrilateral contours, the page detection should select the contour with the largest area.
**Validates: Requirements 3.4**

### Property 9: Aspect ratio preservation
*For any* detected page boundary with aspect ratio R, the perspective-corrected output should have an aspect ratio within 5% of R.
**Validates: Requirements 4.2**

### Property 10: Edge parallelism
*For any* perspective-corrected image, the detected page edges should be parallel to the image borders within 2 degrees tolerance.
**Validates: Requirements 4.3**

### Property 11: Dewarping application
*For any* page image with detected text line curvature above threshold, the dewarping function should be applied and curvature parameters should be computed.
**Validates: Requirements 5.1, 5.2**

### Property 12: Text line straightness improvement
*For any* page image with curved text lines, the straightness metric (measured as standard deviation of text line angles) should decrease after dewarping.
**Validates: Requirements 5.3**

### Property 13: Test mode filtering
*For any* test mode execution with specified sample indices S, only images with indices in S should be processed.
**Validates: Requirements 6.1**

### Property 14: Configuration isolation
*For any* parameter modifications in test mode, the batch processing configuration should remain unchanged.
**Validates: Requirements 6.4**

### Property 15: Log completeness
*For any* batch processing run, the log should contain: total image count, success count, failure count, and total processing time.
**Validates: Requirements 7.1, 7.2, 7.4**

### Property 16: Error logging detail
*For any* processing error, the log should contain both the filename and a non-empty error description.
**Validates: Requirements 7.3**

### Property 17: Resolution preservation
*For any* input page image with dimensions (W, H), the output image dimensions (W', H') should satisfy W' ≥ W/2 and H' ≥ H (accounting for splitting).
**Validates: Requirements 8.1**

### Property 18: Quality metric preservation
*For any* input page image, the output image should maintain sharpness (edge strength) and contrast within 90% of the input values.
**Validates: Requirements 8.2**

### Property 19: Color preservation
*For any* color input page image (3 channels), the output image should also be color (3 channels) with color information preserved.
**Validates: Requirements 8.4**

### Property 20: Function interface consistency
*For any* processing function in the pipeline, it should accept a numpy array as input and return a numpy array as output.
**Validates: Requirements 9.2**

## Error Handling

### Error Categories

1. **File I/O Errors**
   - Missing input files
   - Permission denied
   - Corrupted image files
   - Disk space issues

2. **Processing Errors**
   - Page boundary detection failure
   - Invalid image dimensions
   - Insufficient image quality
   - Dewarping algorithm failure

3. **Configuration Errors**
   - Invalid parameters
   - Missing directories
   - Incompatible settings

### Error Handling Strategy

**Graceful Degradation**:
- If page detection fails, log warning and save original image
- If dewarping fails, save perspective-corrected image
- If perspective correction fails, save preprocessed image
- Continue batch processing even when individual images fail

**Error Logging**:
```python
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class PageDetectionError(ProcessingError):
    """Raised when page boundary cannot be detected."""
    pass

class PerspectiveError(ProcessingError):
    """Raised when perspective transform fails."""
    pass

class DewarpError(ProcessingError):
    """Raised when dewarping algorithm fails."""
    pass
```

**Error Recovery**:
- Retry with relaxed parameters (e.g., lower edge detection thresholds)
- Fall back to simpler algorithms (e.g., skip dewarping if it fails)
- Provide detailed error messages for debugging

**Validation Checks**:
- Validate input image format and dimensions before processing
- Check for sufficient image quality (resolution, contrast)
- Verify detected boundaries are reasonable (size, shape)
- Validate output image quality before saving

## Testing Strategy

### Unit Testing

Unit tests will verify individual components work correctly with specific examples:

**Preprocessing Tests**:
- Test grayscale conversion with color images
- Test noise reduction doesn't over-blur
- Test contrast enhancement improves edge detection

**Page Detection Tests**:
- Test corner detection on images with clear boundaries
- Test handling of images with no detectable page
- Test contour selection with multiple candidates
- Test corner ordering consistency

**Perspective Correction Tests**:
- Test transform calculation with known corner points
- Test aspect ratio preservation with various page shapes
- Test handling of extreme perspective angles

**Dewarping Tests**:
- Test text line detection on straight and curved pages
- Test curvature measurement accuracy
- Test dewarping doesn't introduce artifacts

**Logging Tests**:
- Test log file creation and formatting
- Test error message recording
- Test statistics calculation

### Property-Based Testing

Property-based tests will verify universal properties hold across many randomly generated inputs using **Hypothesis** (Python's property-based testing library).

**Configuration**:
- Each property test will run a minimum of 100 iterations
- Tests will use Hypothesis strategies to generate diverse test cases
- Each test will be tagged with the format: `# Feature: cookbook-dewarp, Property N: <property_text>`

**Test Generators**:

```python
from hypothesis import given, strategies as st
import numpy as np

# Strategy for generating test images
@st.composite
def test_images(draw):
    """Generate random test images with various properties."""
    width = draw(st.integers(min_value=800, max_value=2000))
    height = draw(st.integers(min_value=600, max_value=1500))
    channels = draw(st.sampled_from([1, 3]))  # Grayscale or color
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

# Strategy for generating page boundaries
@st.composite
def page_boundaries(draw, image_shape):
    """Generate valid page boundary corner points."""
    h, w = image_shape[:2]
    # Generate 4 corners within image bounds
    corners = []
    for _ in range(4):
        x = draw(st.integers(min_value=0, max_value=w-1))
        y = draw(st.integers(min_value=0, max_value=h-1))
        corners.append([x, y])
    return np.array(corners, dtype=np.float32)

# Strategy for generating filenames
@st.composite
def image_filenames(draw):
    """Generate valid image filenames."""
    name = draw(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), 
                       min_size=1, max_size=20))
    return f"{name}.jpg"
```

**Property Test Examples**:

```python
# Feature: cookbook-dewarp, Property 2: Filename preservation
@given(filename=image_filenames())
def test_filename_preservation(filename):
    """For any input filename, output should preserve it."""
    # Test implementation
    pass

# Feature: cookbook-dewarp, Property 6: Aspect ratio preservation
@given(corners=page_boundaries(image_shape=(1000, 1000)))
def test_aspect_ratio_preservation(corners):
    """For any page boundary, aspect ratio should be preserved within 5%."""
    # Test implementation
    pass

# Feature: cookbook-dewarp, Property 14: Resolution preservation
@given(image=test_images())
def test_resolution_preservation(image):
    """For any input image, output dimensions should be >= input dimensions."""
    # Test implementation
    pass
```

### Integration Testing

Integration tests will verify the complete pipeline works end-to-end:

- Test full pipeline on sample cookbook images
- Test batch processing with mixed valid/invalid images
- Test test mode vs batch mode behavior
- Test logging and error reporting across full runs

### Test Data

**Sample Images**:
- Use pages 1, 50, 100, 150, 200 as representative samples
- Create synthetic test images with known distortions
- Include edge cases: very curved pages, poor lighting, multiple objects

**Expected Outputs**:
- Manually corrected reference images for comparison
- Known-good processing results for regression testing

### Testing Tools

- **pytest**: Test framework
- **Hypothesis**: Property-based testing
- **pytest-cov**: Code coverage measurement
- **OpenCV test utilities**: Image comparison functions

### Success Criteria

- All unit tests pass
- All property tests pass (100+ iterations each)
- Code coverage > 80%
- Integration tests pass on all sample images
- Manual QA confirms visual quality on sample outputs
