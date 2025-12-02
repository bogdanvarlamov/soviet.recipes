# Implementation Plan

- [ ] 1. Set up project structure and utilities
  - Create `phase_1/post_processing/scripts/` directory
  - Create `phase_1/post_processing/test_outputs/` directory
  - Create `utils.py` with basic image loading and saving functions
  - Set up logging configuration
  - _Requirements: 1.5, 9.1_

- [ ] 2. Implement basic test sample script
  - Create `test_samples.py` with command-line argument parsing
  - Implement function to load specific sample images by page number
  - Implement side-by-side visualization of original images
  - Test with 5 sample images: pages 2, 50, 100, 150, 200 (two-page spreads, skipping covers initially)
  - _Requirements: 6.1, 6.2_

- [ ] 3. Implement preprocessing module
  - Create preprocessing functions in `utils.py`: `load_image()`, `preprocess()`, `enhance_contrast()`
  - Implement grayscale conversion
  - Implement noise reduction (Gaussian blur)
  - Implement contrast enhancement (CLAHE or histogram equalization)
  - Test preprocessing on sample images and visualize results
  - _Requirements: 9.1, 9.2_

- [ ]* 3.1 Write property test for preprocessing
  - **Property 20: Function interface consistency**
  - **Validates: Requirements 9.2**

- [ ] 4. Implement spine detection and splitting
  - Create `detect_spread_type()` function to determine single vs two-page
  - Create `detect_spine_location()` using vertical line detection (Hough Transform) and/or darkness analysis
  - Create `detect_spine_angle()` to measure spine orientation
  - Create `split_along_spine()` to split image into left and right pages
  - Add visualization to draw detected spine line on test samples
  - Test on 5 sample two-page spreads and verify splitting works correctly
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 4.1 Write property test for two-page spread splitting
  - **Property 2: Two-page spread splitting**
  - **Validates: Requirements 2.3**

- [ ]* 4.2 Write property test for spine detection consistency
  - **Property 4: Spine detection consistency**
  - **Validates: Requirements 2.2**

- [ ] 5. Checkpoint - Validate spine detection and splitting
  - Run test_samples.py on 5 two-page spread images
  - Visually inspect spine detection accuracy and split quality
  - Ensure all tests pass, ask the user if questions arise or adjustments are needed

- [ ] 6. Implement page boundary detection
  - Create `detect_page_boundary()` function in `utils.py`
  - Implement edge detection using Canny algorithm
  - Implement contour finding and filtering
  - Create `find_largest_contour()` to select page boundary
  - Create `order_points()` to consistently order corners (TL, TR, BR, BL)
  - Create `validate_quadrilateral()` to check boundary validity
  - Add visualization to draw detected boundaries on individual page images
  - Test on split page images from previous step
  - _Requirements: 3.1, 3.3, 3.4_

- [ ]* 6.1 Write property test for page boundary structure
  - **Property 7: Page boundary structure**
  - **Validates: Requirements 3.1, 3.3**

- [ ]* 6.2 Write property test for largest contour selection
  - **Property 8: Largest contour selection**
  - **Validates: Requirements 3.4**

- [ ] 7. Implement perspective correction
  - Create `apply_perspective_warp()` function in `utils.py`
  - Implement `compute_target_dimensions()` to calculate output size preserving aspect ratio
  - Implement `calculate_perspective_transform()` to compute transform matrix
  - Apply perspective transformation using cv2.warpPerspective
  - Test on individual page images and visualize before/after
  - _Requirements: 4.1, 4.2, 4.3_

- [ ]* 7.1 Write property test for aspect ratio preservation
  - **Property 9: Aspect ratio preservation**
  - **Validates: Requirements 4.2**

- [ ]* 7.2 Write property test for edge parallelism
  - **Property 10: Edge parallelism**
  - **Validates: Requirements 4.3**

- [ ] 8. Checkpoint - Validate full pipeline through perspective correction
  - Run test_samples.py on 5 sample images through: spine detection → splitting → boundary detection → perspective correction
  - Visually inspect results at each stage
  - Ensure all tests pass, ask the user if questions arise or adjustments are needed

- [ ] 9. Implement cylindrical dewarping
  - Create `apply_cylindrical_dewarp()` function in `utils.py`
  - Implement `detect_text_lines()` to identify horizontal text regions using Hough transform or projection
  - Implement `measure_curvature()` to analyze curve patterns in text lines
  - Implement `create_dewarp_mesh()` to generate correction grid
  - Apply mesh-based warping to straighten curved pages
  - Test on sample images with visible curvature
  - _Requirements: 5.1, 5.2, 5.3_

- [ ]* 9.1 Write property test for dewarping application
  - **Property 11: Dewarping application**
  - **Validates: Requirements 5.1, 5.2**

- [ ]* 9.2 Write property test for text line straightness improvement
  - **Property 12: Text line straightness improvement**
  - **Validates: Requirements 5.3**

- [ ] 10. Implement post-processing
  - Create `postprocess()` function in `utils.py`
  - Implement `sharpen_image()` using unsharp mask or kernel convolution
  - Implement `adjust_contrast()` for final contrast optimization
  - Implement `validate_quality()` to check output meets quality thresholds
  - Test on sample images
  - _Requirements: 8.2_

- [ ]* 10.1 Write property test for quality metric preservation
  - **Property 18: Quality metric preservation**
  - **Validates: Requirements 8.2**

- [ ] 11. Integrate full pipeline in test script
  - Update `test_samples.py` to run complete pipeline: spine detect → split → preprocess → detect → perspective → dewarp → postprocess
  - Add side-by-side comparison visualization (original vs split pages vs final corrected)
  - Save test outputs to `test_outputs/` directory with descriptive filenames (e.g., pages-50_left.jpg, pages-50_right.jpg)
  - Add timing information for each processing stage
  - _Requirements: 6.2, 6.3_

- [ ] 12. Checkpoint - Validate complete pipeline on test samples
  - Run complete pipeline on 5 two-page spread sample images
  - Visually inspect final corrected images for both left and right pages
  - Ensure all tests pass, ask the user if questions arise or adjustments are needed

- [ ] 13. Add single-page cover handling
  - Implement logic in `detect_spread_type()` to identify single-page images (covers)
  - Test detection on pages 1 and 224 (front and back covers)
  - Ensure single-page images bypass spine detection and splitting
  - Process single-page images through the rest of the pipeline
  - _Requirements: 2.1, 2.5_

- [ ]* 13.1 Write property test for single-page preservation
  - **Property 3: Single-page preservation**
  - **Validates: Requirements 2.5**

- [ ] 14. Implement error handling and graceful degradation
  - Add try-except blocks around each processing stage
  - Implement fallback behavior: if spine detection fails on two-page spread, process as single image
  - Implement fallback behavior: if dewarping fails, save perspective-corrected image
  - Implement fallback behavior: if perspective fails, save preprocessed image
  - Create custom exception classes: `SpineDetectionError`, `PageDetectionError`, `PerspectiveError`, `DewarpError`
  - Test error handling with intentionally problematic images
  - _Requirements: 3.2, 1.4_

- [ ]* 14.1 Write property test for error resilience
  - **Property 6: Error resilience**
  - **Validates: Requirements 1.4**

- [ ] 15. Implement batch processing script
  - Create `dewarp.py` main script
  - Implement `main()` with command-line argument parsing (input dir, output dir, options)
  - Implement `setup_directories()` to create output directories if needed
  - Implement `process_single_image()` to process one image through complete pipeline
  - Implement `process_batch()` to iterate through all images in input directory
  - Add progress bar or status updates during batch processing
  - Handle filename generation for split pages (e.g., pages-50_left.jpg, pages-50_right.jpg)
  - _Requirements: 1.1, 1.2, 1.5_

- [ ]* 15.1 Write property test for batch processing completeness
  - **Property 1: Batch processing completeness**
  - **Validates: Requirements 1.1, 7.1**

- [ ]* 15.2 Write property test for filename preservation with page suffix
  - **Property 5: Filename preservation with page suffix**
  - **Validates: Requirements 1.2**

- [ ] 16. Implement comprehensive logging
  - Create `init_logger()` function to set up logging to file and console
  - Implement `log_processing_result()` to record individual image results
  - Implement `generate_summary_report()` to create final statistics
  - Log: total images, two-page vs single-page counts, total pages generated, successes, failures, processing time, errors with filenames
  - Save log to `phase_1/post_processing/processing_log.txt`
  - _Requirements: 1.3, 7.1, 7.2, 7.3, 7.4_

- [ ]* 16.1 Write property test for log completeness
  - **Property 15: Log completeness**
  - **Validates: Requirements 7.1, 7.2, 7.4**

- [ ]* 16.2 Write property test for error logging detail
  - **Property 16: Error logging detail**
  - **Validates: Requirements 7.3**

- [ ] 17. Implement configuration management
  - Create `ProcessingConfig` dataclass in `utils.py`
  - Add configuration parameters: edge detection thresholds, spine detection sensitivity, dewarping strength, output quality
  - Implement configuration loading from command-line arguments or config file
  - Ensure test mode and batch mode use separate configurations
  - _Requirements: 6.4_

- [ ]* 17.1 Write property test for configuration isolation
  - **Property 14: Configuration isolation**
  - **Validates: Requirements 6.4**

- [ ] 18. Implement quality preservation features
  - Add resolution preservation logic to ensure output dimensions are appropriate (accounting for splitting)
  - Implement high-quality JPEG saving (quality=95) or PNG for lossless
  - Add color preservation: detect if input is color, maintain color in output
  - Implement sharpness and contrast measurement functions
  - _Requirements: 8.1, 8.3, 8.4_

- [ ]* 18.1 Write property test for resolution preservation
  - **Property 17: Resolution preservation**
  - **Validates: Requirements 8.1**

- [ ]* 18.2 Write property test for color preservation
  - **Property 19: Color preservation**
  - **Validates: Requirements 8.4**

- [ ] 19. Final checkpoint - Test complete system
  - Run test_samples.py on expanded sample set including covers (pages 1, 2, 50, 100, 150, 200, 224)
  - Verify spine detection works on two-page spreads
  - Verify single-page processing works on covers
  - Verify all processing stages work correctly
  - Verify error handling and logging work as expected
  - Ensure all tests pass, ask the user if questions arise or ready for full batch processing

- [ ] 20. Run batch processing on all images
  - Execute `dewarp.py` on all 224 images in `phase_1/cookbook_images/`
  - Monitor progress and check for errors
  - Review processing log for any failures
  - Verify output count: should be ~446 pages (222 two-page spreads × 2 + 2 single covers)
  - Perform visual QA on sample outputs from processed_images directory
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 21. Final validation and documentation
  - Review processing log statistics
  - Perform visual QA on random sample of corrected images (both left and right pages)
  - Document any images that need manual review or reprocessing
  - Update README.md with usage instructions and results summary
  - _Requirements: 7.1, 7.2, 7.3, 7.4_
