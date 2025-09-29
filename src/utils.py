"""
Utility functions for the reverse image search application.
"""
import os
import hashlib
from typing import List, Tuple
from PIL import Image
import streamlit as st

def get_supported_image_extensions() -> List[str]:
    """Get list of supported image file extensions."""
    return ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image format."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in get_supported_image_extensions()

def get_image_files_from_directory(directory: str) -> List[str]:
    """Get all image files from a directory recursively."""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path):
                image_files.append(file_path)
    return image_files

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        st.error(f"Error calculating hash for {file_path}: {str(e)}")
        return ""

def resize_image_for_display(image: Image.Image, max_size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Resize image for display while maintaining aspect ratio."""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def validate_image_file(file_path: str) -> bool:
    """Validate if an image file can be opened and processed."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def create_directory_if_not_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename