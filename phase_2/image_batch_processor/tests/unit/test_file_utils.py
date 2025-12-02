"""Unit tests for file utility functions."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from utils.file_utils import (
    discover_images,
    generate_output_filename,
    ensure_output_directory,
    save_text_to_file,
)


class TestDiscoverImages:
    """Tests for discover_images function."""
    
    def test_discovers_images_with_supported_extensions(self, tmp_path):
        """Test that only files with supported extensions are discovered."""
        # Create test files
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()
        (tmp_path / "image3.jpeg").touch()
        (tmp_path / "document.txt").touch()
        (tmp_path / "data.json").touch()
        
        result = discover_images(tmp_path, [".jpg", ".png", ".jpeg"])
        
        assert len(result) == 3
        assert all(p.suffix.lower() in [".jpg", ".png", ".jpeg"] for p in result)
    
    def test_returns_sorted_list(self, tmp_path):
        """Test that results are sorted by filename."""
        # Create files in non-alphabetical order
        (tmp_path / "zebra.jpg").touch()
        (tmp_path / "apple.jpg").touch()
        (tmp_path / "middle.jpg").touch()
        
        result = discover_images(tmp_path, [".jpg"])
        
        assert len(result) == 3
        assert result[0].name == "apple.jpg"
        assert result[1].name == "middle.jpg"
        assert result[2].name == "zebra.jpg"
    
    def test_case_insensitive_extension_matching(self, tmp_path):
        """Test that extension matching is case-insensitive."""
        (tmp_path / "image1.JPG").touch()
        (tmp_path / "image2.Png").touch()
        (tmp_path / "image3.JPEG").touch()
        
        result = discover_images(tmp_path, [".jpg", ".png", ".jpeg"])
        
        assert len(result) == 3
    
    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Test that empty directory returns empty list."""
        result = discover_images(tmp_path, [".jpg", ".png"])
        
        assert result == []
    
    def test_no_matching_files_returns_empty_list(self, tmp_path):
        """Test that directory with no matching files returns empty list."""
        (tmp_path / "document.txt").touch()
        (tmp_path / "data.json").touch()
        
        result = discover_images(tmp_path, [".jpg", ".png"])
        
        assert result == []
    
    def test_raises_error_for_nonexistent_directory(self):
        """Test that ValueError is raised for non-existent directory."""
        nonexistent = Path("/nonexistent/directory")
        
        with pytest.raises(ValueError, match="Directory does not exist"):
            discover_images(nonexistent, [".jpg"])
    
    def test_raises_error_for_file_path(self, tmp_path):
        """Test that ValueError is raised when path is a file, not directory."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        with pytest.raises(ValueError, match="Path is not a directory"):
            discover_images(file_path, [".jpg"])
    
    def test_ignores_subdirectories(self, tmp_path):
        """Test that subdirectories are not included in results."""
        (tmp_path / "image.jpg").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.jpg").touch()
        
        result = discover_images(tmp_path, [".jpg"])
        
        # Should only find the top-level image, not the nested one
        assert len(result) == 1
        assert result[0].name == "image.jpg"


class TestGenerateOutputFilename:
    """Tests for generate_output_filename function."""
    
    def test_replaces_extension_with_txt(self):
        """Test that image extension is replaced with .txt."""
        image_path = Path("/dir/photo.jpg")
        
        result = generate_output_filename(image_path)
        
        assert result == "photo.txt"
    
    def test_handles_different_extensions(self):
        """Test that various image extensions are handled correctly."""
        test_cases = [
            (Path("/dir/image.png"), "image.txt"),
            (Path("/dir/scan.jpeg"), "scan.txt"),
            (Path("/dir/pic.tiff"), "pic.txt"),
            (Path("/dir/photo.bmp"), "photo.txt"),
        ]
        
        for image_path, expected in test_cases:
            result = generate_output_filename(image_path)
            assert result == expected
    
    def test_custom_output_extension(self):
        """Test that custom output extension can be specified."""
        image_path = Path("/dir/photo.jpg")
        
        result = generate_output_filename(image_path, ".md")
        
        assert result == "photo.md"
    
    def test_adds_dot_to_extension_if_missing(self):
        """Test that dot is added to extension if not provided."""
        image_path = Path("/dir/photo.jpg")
        
        result = generate_output_filename(image_path, "txt")
        
        assert result == "photo.txt"
    
    def test_preserves_filename_with_multiple_dots(self):
        """Test that filenames with multiple dots are handled correctly."""
        image_path = Path("/dir/my.photo.v2.jpg")
        
        result = generate_output_filename(image_path)
        
        assert result == "my.photo.v2.txt"
    
    def test_handles_filename_without_extension(self):
        """Test that files without extension are handled."""
        image_path = Path("/dir/imagefile")
        
        result = generate_output_filename(image_path)
        
        assert result == "imagefile.txt"


class TestEnsureOutputDirectory:
    """Tests for ensure_output_directory function."""
    
    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        output_dir = tmp_path / "output"
        
        assert not output_dir.exists()
        
        ensure_output_directory(output_dir)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_creates_nested_directories(self, tmp_path):
        """Test that parent directories are created as needed."""
        output_dir = tmp_path / "level1" / "level2" / "level3"
        
        assert not output_dir.exists()
        
        ensure_output_directory(output_dir)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_does_nothing_if_directory_exists(self, tmp_path):
        """Test that existing directory is left unchanged."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        
        # Create a file in the directory to verify it's not modified
        test_file = output_dir / "test.txt"
        test_file.write_text("content")
        
        ensure_output_directory(output_dir)
        
        assert output_dir.exists()
        assert test_file.exists()
        assert test_file.read_text() == "content"
    
    def test_raises_error_if_path_is_file(self, tmp_path):
        """Test that ValueError is raised if path exists as a file."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        
        with pytest.raises(ValueError, match="Path exists but is not a directory"):
            ensure_output_directory(file_path)


class TestSaveTextToFile:
    """Tests for save_text_to_file function."""
    
    def test_saves_text_to_file(self, tmp_path):
        """Test that text is correctly saved to file."""
        output_path = tmp_path / "output.txt"
        text = "Hello, world!"
        
        save_text_to_file(text, output_path)
        
        assert output_path.exists()
        assert output_path.read_text(encoding='utf-8') == text
    
    def test_overwrites_existing_file(self, tmp_path):
        """Test that existing file is overwritten."""
        output_path = tmp_path / "output.txt"
        output_path.write_text("old content")
        
        new_text = "new content"
        save_text_to_file(new_text, output_path)
        
        assert output_path.read_text(encoding='utf-8') == new_text
    
    def test_handles_multiline_text(self, tmp_path):
        """Test that multiline text is saved correctly."""
        output_path = tmp_path / "output.txt"
        text = "Line 1\nLine 2\nLine 3"
        
        save_text_to_file(text, output_path)
        
        assert output_path.read_text(encoding='utf-8') == text
    
    def test_handles_unicode_text(self, tmp_path):
        """Test that unicode characters are saved correctly."""
        output_path = tmp_path / "output.txt"
        text = "Hello 世界 🌍 Привет"
        
        save_text_to_file(text, output_path)
        
        assert output_path.read_text(encoding='utf-8') == text
    
    def test_handles_empty_text(self, tmp_path):
        """Test that empty text creates an empty file."""
        output_path = tmp_path / "output.txt"
        
        save_text_to_file("", output_path)
        
        assert output_path.exists()
        assert output_path.read_text(encoding='utf-8') == ""
