# Implementation Plan

- [x] 1. Set up project structure and base abstractions




















  - Create directory structure for engines, core, flow, config, and utils modules under the phase_2 dir from root of the project
  - Define custom exception classes (BatchProcessorError, ExtractionError, ConfigurationError, ValidationError)
  - Set up logging utilities with configurable log levels
  - Install required dependencies (crewai, pydantic, hypothesis, pytest)
  - _Requirements: 8.5_

- [x] 2. Implement ExtractionEngine interface and data models









  - Define ExtractionEngine abstract base class with extract_text and validate_config methods
  - Create ProcessingResult dataclass with image_path, success, output_path, error, attempts, processing_time fields
  - Create BatchReport dataclass with summary statistics and success_rate method
  - _Requirements: 2.1, 2.4_

- [x] 3. Implement configuration system





  - Create base EngineConfig model using Pydantic
  - Create DoclingConfig, LLMConfig, and APIConfig models with engine-specific settings
  - Create BatchProcessorConfig model with validation for image_dir, output_dir, engine_type, max_retries
  - _Requirements: 4.5, 7.3_

- [x] 4. Implement EngineFactory





  - Create EngineFactory class with create_engine static method
  - Implement engine type mapping (docling -> DoclingEngine, llm -> LLMEngine, api -> APIEngine)
  - Add validation to raise ValueError for unsupported engine types
  - _Requirements: 3.5_

- [ ]* 4.1 Write property test for EngineFactory
  - **Property 6: Engine factory correctness**
  - **Validates: Requirements 3.5**

- [x] 5. Implement DoclingEngine referncing phase_2/docling dir files for how it works





  - Create DoclingEngine class implementing ExtractionEngine interface
  - Implement extract_text method using Docling library
  - Implement validate_config method to check Docling configuration
  - Handle ExtractionError for failed extractions
  - _Requirements: 3.1_

- [ ] 6. Implement LLMEngine
  - Create LLMEngine class implementing ExtractionEngine interface
  - Implement extract_text method using LLM for image-to-text
  - Implement validate_config method to verify model availability
  - Handle ExtractionError for failed extractions
  - _Requirements: 3.2_

- [ ] 7. Implement APIEngine
  - Create APIEngine class implementing ExtractionEngine interface
  - Implement extract_text method with HTTP API calls
  - Implement validate_config method to check API connectivity
  - Handle ExtractionError for network and API failures
  - _Requirements: 3.3_

- [ ]* 7.1 Write unit tests for engine implementations
  - Test each engine's extract_text method with mocked dependencies
  - Test validate_config for valid and invalid configurations
  - Test error handling and ExtractionError raising
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 8. Implement file utilities





  - Create function to discover image files in directory with supported extensions
  - Create function to generate output filename from image filename
  - Create function to ensure output directory exists (create if needed)
  - Create function to save text content to file
  - _Requirements: 1.1, 1.4, 6.2, 7.4_

- [ ]* 8.1 Write property test for filename transformation
  - **Property 4: Filename transformation consistency**
  - **Validates: Requirements 1.4, 6.2, 6.3**

- [ ]* 8.2 Write property test for image discovery
  - **Property 1: Image discovery completeness**
  - **Validates: Requirements 1.1**

- [ ]* 8.3 Write property test for output directory creation
  - **Property 14: Output directory creation**
  - **Validates: Requirements 7.4**

- [x] 9. Implement BatchProcessor core logic






  - Create BatchProcessor class with engine, output_dir, max_retries, and logger parameters
  - Implement process_batch method to iterate through all images in directory
  - Implement process_single_image method with retry logic and exponential backoff
  - Implement _save_text private method to write extracted text to output files
  - Add logging for processing start, per-image status, errors, and completion summary
  - _Requirements: 1.2, 4.1, 4.2, 6.1, 8.1, 8.2, 8.3, 8.4_

- [ ]* 9.1 Write property test for processing completeness
  - **Property 2: Processing completeness**
  - **Validates: Requirements 1.2**

- [ ]* 9.2 Write property test for one-to-one output mapping
  - **Property 3: One-to-one output mapping**
  - **Validates: Requirements 1.3**

- [ ]* 9.3 Write property test for retry behavior
  - **Property 8: Retry behavior**
  - **Validates: Requirements 4.2, 4.5**

- [ ]* 9.4 Write property test for error isolation
  - **Property 7: Error isolation**
  - **Validates: Requirements 4.3**

- [ ]* 9.5 Write property test for result reporting accuracy
  - **Property 9: Result reporting accuracy**
  - **Validates: Requirements 1.5, 4.4**

- [ ]* 9.6 Write property test for output file creation
  - **Property 10: Output file creation**
  - **Validates: Requirements 6.1**

- [ ]* 9.7 Write property test for output directory adherence
  - **Property 11: Output directory adherence**
  - **Validates: Requirements 6.4**

- [ ]* 9.8 Write property test for file overwrite idempotence
  - **Property 12: File overwrite idempotence**
  - **Validates: Requirements 6.5**

- [ ]* 9.9 Write property test for error logging completeness
  - **Property 16: Error logging completeness**
  - **Validates: Requirements 8.3**

- [ ] 10. Implement input validation
  - Create validation function to check directory exists
  - Create validation function to check directory contains at least one image
  - Create validation function to check engine type is valid
  - Integrate validation into BatchProcessor initialization
  - Raise ValidationError for invalid inputs before processing begins
  - _Requirements: 7.1, 7.2, 7.3_

- [ ]* 10.1 Write property test for input validation completeness
  - **Property 13: Input validation completeness**
  - **Validates: Requirements 7.1, 7.2, 7.3**

- [ ] 11. Implement engine validation enforcement
  - Add engine validation call in BatchProcessor before processing begins
  - Call engine.validate_config() and raise ConfigurationError if validation fails
  - Log validation results
  - _Requirements: 7.5_

- [ ]* 11.1 Write property test for engine validation enforcement
  - **Property 15: Engine validation enforcement**
  - **Validates: Requirements 7.5**

- [ ]* 11.2 Write property test for engine consistency within batch
  - **Property 5: Engine consistency within batch**
  - **Validates: Requirements 2.5, 3.4**

- [ ] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement CrewAI Flow state model




  - Create BatchProcessorState Pydantic model with image_dir, output_dir, engine_type, engine_config fields
  - Add tracking fields: total_images, processed_images, successful, failed, results
  - _Requirements: 5.1_

- [x] 14. Implement ImageBatchProcessorFlow





  - Create ImageBatchProcessorFlow class extending Flow[BatchProcessorState]
  - Implement initialize_workflow method decorated with @start() to validate inputs
  - Implement create_engine method decorated with @listen(initialize_workflow) to instantiate engine via factory
  - Implement discover_images method decorated with @listen(create_engine) to find all images
  - Implement process_images method decorated with @listen(discover_images) to run BatchProcessor
  - Implement generate_report method decorated with @listen(process_images) to create BatchReport
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 14.1 Write integration test for CrewAI Flow
  - Test full workflow execution from start to finish
  - Test state transitions between flow stages
  - Test that flow correctly coordinates all components
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 15. Create CLI interface
  - Create command-line interface to run batch processor
  - Accept arguments for image_dir, output_dir, engine_type, and config file path
  - Load configuration from file or command-line arguments
  - Execute ImageBatchProcessorFlow and display results
  - _Requirements: 1.1, 3.5_

- [ ]* 15.1 Write integration test for end-to-end processing
  - Test processing a small batch of real images with each engine type
  - Verify output files are created correctly
  - Verify batch report accuracy
  - Test mixed success/failure scenarios
  - _Requirements: 1.3, 4.4, 6.1_

- [ ] 16. Create example configurations
  - Create example configuration file for Docling engine
  - Create example configuration file for LLM engine
  - Create example configuration file for API engine
  - Add documentation comments explaining each configuration option
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 17. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
