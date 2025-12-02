"""File handling utilities for the image batch processor."""

from pathlib import Path
from typing import List


def discover_images(directory: Path, supported_extensions: List[str]) -> List[Path]:
    """
    Discover all image files in a directory with supported extensions.
    
    Args:
        directory: Path to the directory to search
        supported_extensions: List of file extensions to include (e.g., ['.jpg', '.png'])
    
    Returns:
        List of Path objects for discovered image files, sorted by name
    
    Raises:
        ValueError: If directory does not exist or is not a directory
    
    Examples:
        >>> discover_images(Path('/images'), ['.jpg', '.png'])
        [Path('/images/photo1.jpg'), Path('/images/photo2.png')]
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Normalize extensions to lowercase for case-insensitive matching
    normalized_extensions = [ext.lower() for ext in supported_extensions]
    
    # Find all files with supported extensions
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in normalized_extensions:
            image_files.append(file_path)
    
    # Sort by filename for consistent ordering
    return sorted(image_files)


def generate_output_filename(image_path: Path, output_extension: str = ".txt") -> str:
    """
    Generate output filename from image filename by replacing the extension.
    
    Args:
        image_path: Path to the source image file
        output_extension: Extension for the output file (default: '.txt')
    
    Returns:
        Output filename (stem + new extension)
    
    Examples:
        >>> generate_output_filename(Path('/dir/photo.jpg'))
        'photo.txt'
        >>> generate_output_filename(Path('/dir/image.png'), '.md')
        'image.md'
    """
    # Ensure output extension starts with a dot
    if not output_extension.startswith('.'):
        output_extension = '.' + output_extension
    
    return image_path.stem + output_extension


def ensure_output_directory(output_dir: Path) -> None:
    """
    Ensure output directory exists, creating it if necessary.
    
    Args:
        output_dir: Path to the output directory
    
    Raises:
        OSError: If directory creation fails
        ValueError: If path exists but is not a directory
    
    Examples:
        >>> ensure_output_directory(Path('/output'))
        # Creates /output if it doesn't exist
    """
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Path exists but is not a directory: {output_dir}")
        # Directory already exists, nothing to do
        return
    
    # Create directory including any necessary parent directories
    output_dir.mkdir(parents=True, exist_ok=True)


def save_text_to_file(text: str, output_path: Path) -> None:
    """
    Save text content to a file.
    
    Args:
        text: Text content to save
        output_path: Path where the file should be saved
    
    Raises:
        OSError: If file writing fails
    
    Examples:
        >>> save_text_to_file("Hello world", Path('/output/file.txt'))
        # Creates /output/file.txt with content "Hello world"
    """
    # Write text to file, overwriting if it exists
    output_path.write_text(text, encoding='utf-8')
