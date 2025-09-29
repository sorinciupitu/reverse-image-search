"""
Search engine for reverse image search functionality.
"""
import os
import tempfile
from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image
import streamlit as st
import logging
from .image_processor import get_image_processor
from .database_manager import get_database_manager
from .utils import validate_image_file, resize_image_for_display

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    """Main search engine for reverse image search."""
    
    def __init__(self):
        """Initialize the search engine."""
        self.image_processor = get_image_processor()
        self.database_manager = get_database_manager()
    
    def search_by_image_file(
        self, 
        image_file, 
        n_results: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar images using an uploaded image file.
        
        Args:
            image_file: Streamlit uploaded file object
            n_results: Number of similar images to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar images with metadata and scores
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Validate the image
                if not validate_image_file(tmp_path):
                    raise ValueError("Invalid image file")
                
                # Extract features from the query image
                query_features = self.image_processor.extract_features(tmp_path)
                
                # Search for similar images
                results = self.database_manager.search_similar_images(
                    query_features, 
                    n_results=n_results
                )
                
                # Filter by similarity threshold
                filtered_results = [
                    result for result in results 
                    if result["similarity_score"] >= similarity_threshold
                ]
                
                return filtered_results
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error searching by image file: {str(e)}")
            raise
    
    def search_by_image_path(
        self, 
        image_path: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar images using a local image path.
        
        Args:
            image_path: Path to the query image
            n_results: Number of similar images to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar images with metadata and scores
        """
        try:
            # Validate the image
            if not validate_image_file(image_path):
                raise ValueError(f"Invalid image file: {image_path}")
            
            # Extract features from the query image
            query_features = self.image_processor.extract_features(image_path)
            
            # Search for similar images
            results = self.database_manager.search_similar_images(
                query_features, 
                n_results=n_results
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result["similarity_score"] >= similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching by image path: {str(e)}")
            raise
    
    def add_images_from_directory(
        self, 
        directory_path: str,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        Add images from a directory to the search database.
        
        Args:
            directory_path: Path to the directory containing images
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation statistics
        """
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            if not os.path.isdir(directory_path):
                raise ValueError(f"Path is not a directory: {directory_path}")
            
            # Add directory embeddings
            stats = self.database_manager.add_directory_embeddings(
                directory_path,
                self.image_processor.extract_features,
                progress_callback
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error adding images from directory: {str(e)}")
            raise
    
    def remove_directory_from_database(self, directory_path: str) -> int:
        """
        Remove all images from a directory from the search database.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Number of images removed
        """
        try:
            deleted_count = self.database_manager.delete_directory_embeddings(directory_path)
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error removing directory from database: {str(e)}")
            raise
    
    def get_database_statistics(self) -> Dict:
        """
        Get statistics about the search database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            return self.database_manager.get_database_stats()
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.database_manager.clear_database()
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def export_database(self, export_path: str = None) -> str:
        """
        Export the database to a portable format.
        
        Args:
            export_path: Optional path for the export file
            
        Returns:
            Path to the exported database file
        """
        try:
            return self.database_manager.export_database(export_path)
        except Exception as e:
            logger.error(f"Error exporting database: {str(e)}")
            raise
    
    def import_database(self, import_path: str, path_mapping: Dict[str, str] = None) -> Dict[str, int]:
        """
        Import database from a portable format.
        
        Args:
            import_path: Path to the exported database file
            path_mapping: Optional mapping for updating file paths
            
        Returns:
            Dictionary with import statistics
        """
        try:
            return self.database_manager.import_database(import_path, path_mapping)
        except Exception as e:
            logger.error(f"Error importing database: {str(e)}")
            raise
    
    def get_export_info(self, import_path: str) -> Dict:
        """
        Get information about an export file without importing it.
        
        Args:
            import_path: Path to the exported database file
            
        Returns:
            Dictionary with export information
        """
        try:
            return self.database_manager.get_export_info(import_path)
        except Exception as e:
            logger.error(f"Error reading export info: {str(e)}")
            return {}
    
    def validate_query_image(self, image_file) -> bool:
        """
        Validate an uploaded query image.
        
        Args:
            image_file: Streamlit uploaded file object
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file size (limit to 10MB)
            if image_file.size > 10 * 1024 * 1024:
                return False
            
            # Try to open as PIL Image
            image = Image.open(image_file)
            image.verify()
            
            return True
            
        except Exception:
            return False
    
    def get_image_preview(self, image_file, max_size=(300, 300)) -> Image.Image:
        """
        Get a preview of an uploaded image.
        
        Args:
            image_file: Streamlit uploaded file object
            max_size: Maximum size for the preview
            
        Returns:
            PIL Image object for preview
        """
        try:
            image = Image.open(image_file).convert('RGB')
            return resize_image_for_display(image, max_size)
        except Exception as e:
            logger.error(f"Error creating image preview: {str(e)}")
            raise

class SearchResult:
    """Class to represent a search result."""
    
    def __init__(self, result_dict: Dict):
        """Initialize search result from dictionary."""
        self.id = result_dict.get("id")
        self.file_path = result_dict.get("file_path")
        self.filename = result_dict.get("filename")
        self.similarity_score = result_dict.get("similarity_score", 0.0)
        self.distance = result_dict.get("distance", 1.0)
        self.metadata = result_dict.get("metadata", {})
    
    def get_display_image(self, max_size=(200, 200)) -> Optional[Image.Image]:
        """Get image for display purposes."""
        try:
            if os.path.exists(self.file_path):
                image = Image.open(self.file_path).convert('RGB')
                return resize_image_for_display(image, max_size)
            return None
        except Exception:
            return None
    
    def is_valid(self) -> bool:
        """Check if the result file still exists."""
        return os.path.exists(self.file_path) if self.file_path else False
    
    def get_file_size_formatted(self) -> str:
        """Get formatted file size."""
        file_size = self.metadata.get("file_size", 0)
        if file_size == 0:
            return "Unknown"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while file_size >= 1024 and i < len(size_names) - 1:
            file_size /= 1024.0
            i += 1
        return f"{file_size:.1f}{size_names[i]}"

# Global instance for reuse
_search_engine = None

def get_search_engine() -> SearchEngine:
    """Get or create a global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine

@st.cache_resource
def load_search_engine() -> SearchEngine:
    """Load search engine with Streamlit caching."""
    return SearchEngine()