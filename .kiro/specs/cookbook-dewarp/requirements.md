# Requirements Document

## Introduction

This document specifies requirements for a cookbook image post-processing system that corrects distortions in scanned cookbook page images. The system processes 224 JPG images containing two-page spreads (and some single-page covers) with curved pages, perspective distortion, and text line curvature to produce corrected individual page images suitable for OCR and digital archival.

## Glossary

- **Dewarp System**: The image processing pipeline that corrects distortions in scanned cookbook images
- **Source Image**: An original distorted JPG image from the cookbook_images directory
- **Two-Page Spread**: A scanned image containing two adjacent pages from an open book
- **Book Spine**: The central binding area separating left and right pages in a two-page spread
- **Page Image**: An individual page extracted from a Source Image (either from splitting or single-page)
- **Corrected Image**: A processed page image with distortions removed, saved to the processed_images directory
- **Page Boundary**: The detected edges of a single cookbook page within an image
- **Perspective Transform**: A geometric transformation that corrects viewing angle distortion
- **Cylindrical Dewarping**: A correction technique for curved page surfaces caused by book binding
- **Processing Log**: A text file recording processing results, errors, and statistics for all images
- **Test Sample**: A subset of images (e.g., pages 2, 50, 100, 150, 200) used for algorithm validation

## Requirements

### Requirement 1

**User Story:** As a digital archivist, I want to batch process all cookbook images, so that I can efficiently correct distortions across the entire collection.

#### Acceptance Criteria

1. WHEN the Dewarp System processes the cookbook_images directory THEN the Dewarp System SHALL process all 224 JPG files
2. WHEN processing multiple images THEN the Dewarp System SHALL save each Corrected Image to the processed_images directory with a filename indicating the source and page side
3. WHEN batch processing completes THEN the Dewarp System SHALL generate a Processing Log containing success/failure status for each Source Image
4. WHEN an individual image processing fails THEN the Dewarp System SHALL continue processing remaining images and record the error in the Processing Log
5. WHEN processing begins THEN the Dewarp System SHALL create the processed_images directory if it does not exist

### Requirement 2

**User Story:** As a digital archivist, I want the system to detect and split two-page spreads, so that each page can be processed independently with correct geometry.

#### Acceptance Criteria

1. WHEN the Dewarp System analyzes a Source Image THEN the Dewarp System SHALL detect whether the image contains a single page or a two-page spread
2. WHEN a two-page spread is detected THEN the Dewarp System SHALL identify the book spine location
3. WHEN the book spine is identified THEN the Dewarp System SHALL split the Source Image into two separate Page Images along the spine
4. WHEN splitting occurs THEN the Dewarp System SHALL handle non-vertical spine angles by detecting the spine orientation
5. WHEN a single-page image is detected THEN the Dewarp System SHALL process it as one Page Image without splitting

### Requirement 3

**User Story:** As a digital archivist, I want the system to detect page boundaries accurately, so that the cookbook content is properly isolated from the background.

#### Acceptance Criteria

1. WHEN the Dewarp System analyzes a Page Image THEN the Dewarp System SHALL detect the four corner points of the Page Boundary
2. WHEN Page Boundary detection fails THEN the Dewarp System SHALL log a warning and skip dewarping for that Page Image
3. WHEN the Page Boundary is detected THEN the Dewarp System SHALL validate that the detected region forms a quadrilateral shape
4. WHEN multiple contours are present THEN the Dewarp System SHALL select the largest quadrilateral contour as the Page Boundary

### Requirement 4

**User Story:** As a digital archivist, I want perspective distortion corrected, so that pages appear as if viewed directly from above.

#### Acceptance Criteria

1. WHEN the Dewarp System has detected a Page Boundary THEN the Dewarp System SHALL apply a Perspective Transform to align the page to a rectangular view
2. WHEN applying Perspective Transform THEN the Dewarp System SHALL preserve the aspect ratio of the original page dimensions
3. WHEN the Perspective Transform is complete THEN the Corrected Image SHALL have all four page edges parallel to the image borders

### Requirement 5

**User Story:** As a digital archivist, I want curved page surfaces flattened, so that text lines appear straight and readable.

#### Acceptance Criteria

1. WHEN the Dewarp System processes a Page Image with curved pages THEN the Dewarp System SHALL apply Cylindrical Dewarping to straighten text lines
2. WHEN Cylindrical Dewarping is applied THEN the Dewarp System SHALL detect text line curvature patterns in the image
3. WHEN text lines are straightened THEN the Corrected Image SHALL show no visible curvature in horizontal text lines
4. WHEN dewarping is complete THEN the Dewarp System SHALL preserve text readability and character shapes

### Requirement 6

**User Story:** As a digital archivist, I want to validate processing quality on test samples, so that I can verify the algorithm works before batch processing.

#### Acceptance Criteria

1. WHEN the Dewarp System runs in test mode THEN the Dewarp System SHALL process only the specified Test Sample images
2. WHEN test mode processing completes THEN the Dewarp System SHALL save Test Sample outputs to the test_outputs directory
3. WHEN processing Test Samples THEN the Dewarp System SHALL display visual comparisons between Source Images and Corrected Images
4. WHEN test mode is used THEN the Dewarp System SHALL allow parameter adjustment without affecting the main batch processing

### Requirement 7

**User Story:** As a digital archivist, I want processing statistics and error reports, so that I can assess the quality and completeness of the batch processing.

#### Acceptance Criteria

1. WHEN batch processing completes THEN the Dewarp System SHALL write a Processing Log containing the total number of images processed
2. WHEN the Processing Log is generated THEN the Processing Log SHALL include the count of successful and failed processing attempts
3. WHEN an error occurs during processing THEN the Processing Log SHALL record the filename and error description
4. WHEN processing finishes THEN the Processing Log SHALL include the total processing time

### Requirement 8

**User Story:** As a digital archivist, I want image quality preserved during processing, so that the corrected images remain suitable for OCR and archival purposes.

#### Acceptance Criteria

1. WHEN the Dewarp System saves a Corrected Image THEN the Dewarp System SHALL maintain image resolution equal to or greater than the original Page Image
2. WHEN processing is applied THEN the Dewarp System SHALL preserve text sharpness and contrast
3. WHEN saving Corrected Images THEN the Dewarp System SHALL use lossless or high-quality compression to prevent quality degradation
4. WHEN color images are processed THEN the Dewarp System SHALL preserve the original color information

### Requirement 9

**User Story:** As a developer, I want modular processing components, so that I can test, debug, and improve individual processing stages independently.

#### Acceptance Criteria

1. WHEN the Dewarp System is implemented THEN the Dewarp System SHALL separate spine detection, preprocessing, page detection, perspective correction, and dewarping into distinct functions
2. WHEN a processing function is called THEN the function SHALL accept an image array as input and return a processed image array as output
3. WHEN processing stages are separated THEN each stage SHALL be testable independently with sample images
4. WHEN utility functions are needed THEN the Dewarp System SHALL provide them in a separate utils module
