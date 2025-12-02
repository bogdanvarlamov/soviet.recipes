# Requirements Document

## Introduction

This document specifies requirements for an image batch processing system that extracts text from images using pluggable extraction engines. The system processes directories of images, converts them to machine-readable text files with one-to-one mapping, and orchestrates the workflow using CrewAI Flows. The architecture emphasizes separation of concerns with an interface-based engine design supporting multiple extraction technologies (Docling, local LLMs, API-based services).

## Glossary

- **Image Batch Processor**: The system that orchestrates batch processing of images to extract text
- **Extraction Engine**: A pluggable component that implements text extraction from images using a specific technology
- **Docling**: A document processing technology used as one extraction engine implementation
- **CrewAI Flows**: The workflow orchestration framework used to manage the processing pipeline
- **One-to-One Mapping**: Each input image produces exactly one output text file with corresponding naming
- **Engine Interface**: The abstract contract that all extraction engine implementations must fulfill

## Requirements

### Requirement 1

**User Story:** As a developer, I want to process a directory of images through a batch workflow, so that I can extract text from all images systematically.

#### Acceptance Criteria

1. WHEN the Image Batch Processor receives an image directory path THEN the system SHALL identify all image files in that directory
2. WHEN processing begins THEN the Image Batch Processor SHALL iterate through each image file sequentially
3. WHEN all images are processed THEN the Image Batch Processor SHALL produce exactly one text file per input image
4. WHEN an image is processed THEN the Image Batch Processor SHALL maintain the original filename with a text extension for the output file
5. THE Image Batch Processor SHALL track processing progress across the entire batch

### Requirement 2

**User Story:** As a developer, I want to use different text extraction technologies interchangeably, so that I can choose the best engine for my use case without changing the workflow code.

#### Acceptance Criteria

1. THE Image Batch Processor SHALL define an Engine Interface that all extraction implementations must implement
2. WHEN an Extraction Engine is invoked THEN the system SHALL call the engine through the Engine Interface
3. WHERE a developer wants to add a new extraction technology THEN the system SHALL allow implementation of the Engine Interface without modifying existing workflow code
4. THE Engine Interface SHALL accept an image file path as input and return extracted text as output
5. WHEN switching between engines THEN the Image Batch Processor SHALL operate identically regardless of which engine implementation is used

### Requirement 3

**User Story:** As a developer, I want multiple extraction engine implementations available, so that I can select the appropriate technology for different scenarios.

#### Acceptance Criteria

1. THE system SHALL provide a Docling-based Extraction Engine implementation
2. THE system SHALL provide an LLM-based Extraction Engine implementation
3. THE system SHALL provide an API-based Extraction Engine implementation
4. WHEN an engine is selected THEN the Image Batch Processor SHALL use that engine for all images in the batch
5. WHERE configuration specifies an engine type THEN the system SHALL instantiate the corresponding engine implementation

### Requirement 4

**User Story:** As a developer, I want the workflow to handle extraction failures gracefully, so that temporary errors do not halt the entire batch process.

#### Acceptance Criteria

1. WHEN an Extraction Engine fails to process an image THEN the system SHALL catch the error and log relevant details
2. IF an extraction fails THEN the Image Batch Processor SHALL retry the operation with the same engine
3. WHEN retry attempts are exhausted THEN the system SHALL mark the image as failed and continue processing remaining images
4. WHEN processing completes THEN the Image Batch Processor SHALL report which images succeeded and which failed
5. THE system SHALL configure maximum retry attempts per image

### Requirement 5

**User Story:** As a developer, I want to use CrewAI Flows to orchestrate the workflow, so that I can leverage its workflow management capabilities.

#### Acceptance Criteria

1. THE Image Batch Processor SHALL implement the workflow using CrewAI Flows framework
2. WHEN the workflow starts THEN the system SHALL use CrewAI Flows to manage state transitions between processing stages
3. THE system SHALL define the batch processing loop as a CrewAI Flow
4. WHEN an image is processed THEN the system SHALL use CrewAI Flows to coordinate between workflow stages
5. THE system SHALL expose workflow control through CrewAI Flows APIs

### Requirement 6

**User Story:** As a developer, I want each extracted text saved to a separate file, so that I can easily map outputs back to source images.

#### Acceptance Criteria

1. WHEN text extraction succeeds THEN the system SHALL write the extracted text to a file
2. THE system SHALL create output filenames by replacing the image extension with a text extension
3. WHEN saving output files THEN the system SHALL preserve the original image filename structure
4. THE system SHALL write output files to a configurable output directory
5. IF an output file already exists THEN the system SHALL overwrite it with new extraction results

### Requirement 7

**User Story:** As a developer, I want the system to validate inputs before processing, so that I can catch configuration errors early.

#### Acceptance Criteria

1. WHEN the Image Batch Processor receives a directory path THEN the system SHALL verify the directory exists
2. WHEN validating inputs THEN the system SHALL confirm at least one image file is present in the directory
3. IF the specified engine type is invalid THEN the system SHALL raise a configuration error before processing begins
4. WHEN the output directory does not exist THEN the system SHALL create it before processing
5. THE system SHALL validate that the selected Extraction Engine is properly configured before starting the batch

### Requirement 8

**User Story:** As a developer, I want clear logging throughout the process, so that I can monitor progress and debug issues.

#### Acceptance Criteria

1. WHEN processing starts THEN the Image Batch Processor SHALL log the total number of images to process
2. WHEN each image is processed THEN the system SHALL log the image filename and processing status
3. WHEN an error occurs THEN the system SHALL log the error message with context about which image failed
4. WHEN processing completes THEN the system SHALL log summary statistics including success count and failure count
5. THE system SHALL support configurable log levels for different verbosity needs
