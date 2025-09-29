"""
Database manager for ChromaDB vector storage and operations.
"""
import os
import uuid
import json
import zipfile
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
import streamlit as st
import logging
from .utils import get_image_files_from_directory, calculate_file_hash, format_file_size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manager for ChromaDB operations and vector storage."""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the ChromaDB database directory
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.collection_name = "image_embeddings"
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll provide embeddings manually
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            st.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_image_embedding(
        self, 
        image_path: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add an image embedding to the database.
        
        Args:
            image_path: Path to the image file
            embedding: Feature vector for the image
            metadata: Additional metadata for the image
            
        Returns:
            Unique ID for the stored embedding
        """
        try:
            # Generate unique ID
            image_id = str(uuid.uuid4())
            
            # Prepare metadata
            file_stats = os.stat(image_path)
            default_metadata = {
                "file_path": image_path,
                "filename": os.path.basename(image_path),
                "directory": os.path.dirname(image_path),
                "file_size": file_stats.st_size,
                "created_at": datetime.now().isoformat(),
                "file_hash": calculate_file_hash(image_path)
            }
            
            if metadata:
                default_metadata.update(metadata)
            
            # Add to collection
            self.collection.add(
                ids=[image_id],
                embeddings=[embedding.tolist()],
                metadatas=[default_metadata]
            )
            
            logger.info(f"Added embedding for {image_path} with ID {image_id}")
            return image_id
            
        except Exception as e:
            logger.error(f"Error adding embedding: {str(e)}")
            raise
    
    def add_directory_embeddings(
        self, 
        directory_path: str, 
        extract_features_func,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        Add embeddings for all images in a directory.
        
        Args:
            directory_path: Path to the directory containing images
            extract_features_func: Function to extract features from images
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with statistics about the operation
        """
        try:
            # Get all image files
            image_files = get_image_files_from_directory(directory_path)
            
            if not image_files:
                return {"total": 0, "processed": 0, "errors": 0}
            
            stats = {"total": len(image_files), "processed": 0, "errors": 0}
            
            for i, image_path in enumerate(image_files):
                try:
                    # Update progress
                    if progress_callback:
                        progress_callback(i + 1, len(image_files), image_path)
                    
                    # Check if image already exists
                    if self.image_exists(image_path):
                        logger.info(f"Skipping existing image: {image_path}")
                        stats["processed"] += 1
                        continue
                    
                    # Extract features
                    features = extract_features_func(image_path)
                    
                    # Add to database
                    self.add_image_embedding(image_path, features)
                    stats["processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    stats["errors"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error adding directory embeddings: {str(e)}")
            raise
    
    def search_similar_images(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar images using vector similarity.
        
        Args:
            query_embedding: Query feature vector
            n_results: Number of similar images to return
            where_filter: Optional metadata filter
            
        Returns:
            List of similar images with metadata and similarity scores
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter,
                include=["metadatas", "distances"]
            )
            
            # Format results
            similar_images = []
            if results["ids"] and results["ids"][0]:
                for i, image_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    similar_images.append({
                        "id": image_id,
                        "file_path": metadata["file_path"],
                        "filename": metadata["filename"],
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "metadata": metadata
                    })
            
            return similar_images
            
        except Exception as e:
            logger.error(f"Error searching similar images: {str(e)}")
            raise
    
    def image_exists(self, image_path: str) -> bool:
        """
        Check if an image already exists in the database.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image exists, False otherwise
        """
        try:
            results = self.collection.get(
                where={"file_path": image_path},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception:
            return False
    
    def delete_image_embedding(self, image_id: str) -> bool:
        """
        Delete an image embedding from the database.
        
        Args:
            image_id: ID of the image embedding to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[image_id])
            logger.info(f"Deleted embedding with ID {image_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            return False
    
    def delete_directory_embeddings(self, directory_path: str) -> int:
        """
        Delete all embeddings for images in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Number of embeddings deleted
        """
        try:
            # Get all embeddings for the directory
            results = self.collection.get(
                where={"directory": directory_path}
            )
            
            if results["ids"]:
                # Delete all embeddings
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(f"Deleted {deleted_count} embeddings for directory {directory_path}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting directory embeddings: {str(e)}")
            raise
    
    def clear_database(self) -> bool:
        """
        Clear all embeddings from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get all embeddings
            results = self.collection.get()
            
            total_images = len(results["ids"])
            
            # Calculate directory statistics
            directories = {}
            total_size = 0
            
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    directory = metadata.get("directory", "Unknown")
                    file_size = metadata.get("file_size", 0)
                    
                    if directory not in directories:
                        directories[directory] = {"count": 0, "size": 0}
                    
                    directories[directory]["count"] += 1
                    directories[directory]["size"] += file_size
                    total_size += file_size
            
            return {
                "total_images": total_images,
                "total_size": total_size,
                "total_size_formatted": format_file_size(total_size),
                "directories": directories,
                "database_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {
                "total_images": 0,
                "total_size": 0,
                "total_size_formatted": "0B",
                "directories": {},
                "database_path": self.db_path
            }
    
    def export_database(self, export_path: str = None) -> str:
        """
        Export the database to a portable format.
        
        Args:
            export_path: Optional path for the export file
            
        Returns:
            Path to the exported database file
        """
        try:
            # Create export filename if not provided
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"database_export_{timestamp}.zip"
            
            # Get all data from the collection
            results = self.collection.get(
                include=["metadatas", "embeddings"]
            )
            
            if not results["ids"]:
                raise ValueError("No data to export - database is empty")
            
            # Prepare export data
            export_data = {
                "version": "1.0",
                "export_timestamp": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "total_images": len(results["ids"]),
                "data": []
            }
            
            # Process each image record
            for i, image_id in enumerate(results["ids"]):
                record = {
                    "id": image_id,
                    "embedding": results["embeddings"][i],
                    "metadata": results["metadatas"][i]
                }
                export_data["data"].append(record)
            
            # Create temporary directory for export
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save data as JSON
                json_path = os.path.join(temp_dir, "database_data.json")
                with open(json_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                # Copy ChromaDB files if they exist
                chroma_files = []
                if os.path.exists(self.db_path):
                    for root, dirs, files in os.walk(self.db_path):
                        for file in files:
                            if file.endswith(('.bin', '.pickle', '.sqlite3')):
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, self.db_path)
                                chroma_files.append((file_path, rel_path))
                
                # Create ZIP archive
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add JSON data
                    zipf.write(json_path, "database_data.json")
                    
                    # Add ChromaDB files
                    for file_path, rel_path in chroma_files:
                        zipf.write(file_path, f"chroma_db/{rel_path}")
                    
                    # Add metadata file
                    metadata = {
                        "export_info": {
                            "version": "1.0",
                            "timestamp": datetime.now().isoformat(),
                            "total_images": len(results["ids"]),
                            "collection_name": self.collection_name
                        }
                    }
                    
                    metadata_json = json.dumps(metadata, indent=2)
                    zipf.writestr("export_metadata.json", metadata_json)
            
            logger.info(f"Database exported successfully to {export_path}")
            return export_path
            
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
            if not os.path.exists(import_path):
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            stats = {"imported": 0, "skipped": 0, "errors": 0}
            
            # Extract and read the ZIP file
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                
                # Read the JSON data
                json_path = os.path.join(temp_dir, "database_data.json")
                if not os.path.exists(json_path):
                    raise ValueError("Invalid export file - missing database_data.json")
                
                with open(json_path, 'r') as f:
                    import_data = json.load(f)
                
                # Validate import data
                if "data" not in import_data:
                    raise ValueError("Invalid export file - missing data section")
                
                logger.info(f"Importing {len(import_data['data'])} records from {import_data.get('export_timestamp', 'unknown')}")
                
                # Process each record
                for record in import_data["data"]:
                    try:
                        image_id = record["id"]
                        embedding = np.array(record["embedding"])
                        metadata = record["metadata"].copy()
                        
                        # Update file paths if mapping provided
                        if path_mapping and "file_path" in metadata:
                            old_path = metadata["file_path"]
                            for old_prefix, new_prefix in path_mapping.items():
                                if old_path.startswith(old_prefix):
                                    new_path = old_path.replace(old_prefix, new_prefix, 1)
                                    metadata["file_path"] = new_path
                                    metadata["directory"] = os.path.dirname(new_path)
                                    break
                        
                        # Check if record already exists
                        existing = self.collection.get(ids=[image_id])
                        if existing["ids"]:
                            stats["skipped"] += 1
                            continue
                        
                        # Add to collection
                        self.collection.add(
                            ids=[image_id],
                            embeddings=[embedding.tolist()],
                            metadatas=[metadata]
                        )
                        
                        stats["imported"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error importing record {record.get('id', 'unknown')}: {str(e)}")
                        stats["errors"] += 1
            
            logger.info(f"Import completed: {stats['imported']} imported, {stats['skipped']} skipped, {stats['errors']} errors")
            return stats
            
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
            if not os.path.exists(import_path):
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    # Extract metadata file
                    try:
                        metadata_content = zipf.read("export_metadata.json")
                        metadata = json.loads(metadata_content.decode('utf-8'))
                        return metadata.get("export_info", {})
                    except KeyError:
                        # Fallback to reading database_data.json
                        data_content = zipf.read("database_data.json")
                        data = json.loads(data_content.decode('utf-8'))
                        return {
                            "version": data.get("version", "unknown"),
                            "timestamp": data.get("export_timestamp", "unknown"),
                            "total_images": data.get("total_images", 0),
                            "collection_name": data.get("collection_name", "unknown")
                        }
        
        except Exception as e:
            logger.error(f"Error reading export info: {str(e)}")
            return {}

# Global instance for reuse
_database_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create a global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

@st.cache_resource
def load_database_manager() -> DatabaseManager:
    """Load database manager with Streamlit caching."""
    return DatabaseManager()