# Docling Document Processing

This project processes scanned cookbook images using Docling with OCR to extract Russian text.

## Setup

1. Create conda environment:
```bash
conda create -n cookbook-processing python=3.11
conda activate cookbook-processing
```

2. Install uv package manager:
```bash
pip install uv
```

3. Initialize uv project and install dependencies:
```bash
uv init
uv add docling easyocr
```

## Configuration

The script uses EasyOCR for optical character recognition with Russian and English language support. Configuration is in `process_documents.py`:

- `INPUT_FILES`: List of image files to process
- `OUTPUT_DIR`: Directory for output files (default: `output/`)

## Usage

```bash
python process_documents.py
```

## Output

For each processed image, the script generates:
- `.md` file: Markdown formatted text
- `.doctags.json` file: Native Docling format preserving all metadata and structure

## Notes

- EasyOCR was chosen over Tesseract due to easier installation on Windows with conda environments
- The script supports both PDF and image formats (JPG, PNG, etc.)
- Russian language detection is configured via `EasyOcrOptions(lang=["ru", "en"])`
