# Product Overview

soviet.recipes is a cookbook digitization project that converts scanned cookbook images into machine-readable text.

## Current Phase

The project is in Phase 2, focused on building an image batch processing system that:
- Extracts text from directories of cookbook images
- Supports multiple extraction engines (Docling, LLM-based, API-based)
- Maintains one-to-one mapping between input images and output text files
- Uses CrewAI Flows for workflow orchestration

## Project Structure

- **Phase 1**: Contains 224 scanned cookbook images (pages-1.jpg through pages-224.jpg)
- **Phase 2**: Active development of the image batch processor system

## Key Goals

- Systematic batch processing of cookbook images
- Pluggable extraction engine architecture for flexibility
- Robust error handling with retry logic
- Clear progress tracking and logging
