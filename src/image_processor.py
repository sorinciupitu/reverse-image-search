"""
Image processing module for feature extraction using EfficientNet-B0.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import streamlit as st
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processor using EfficientNet-B0 for feature extraction."""
    
    def __init__(self, model_name: str = 'efficientnet_b0', device: Optional[str] = None):
        """
        Initialize the image processor.
        
        Args:
            model_name: Name of the model to use (default: efficientnet_b0)
            device: Device to run the model on (auto-detect if None)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._initialize_model()
        self._initialize_transform()
    
    def _initialize_model(self) -> None:
        """Initialize the EfficientNet-B0 model."""
        try:
            # Load pre-trained EfficientNet-B0 model
            self.model = timm.create_model(
                self.model_name, 
                pretrained=True, 
                num_classes=0  # Remove classification head to get features
            )
            self.model.eval()
            self.model.to(self.device)
            
            # Get the feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.model(dummy_input)
                self.feature_dim = dummy_output.shape[1]
            
            logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            logger.info(f"Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            raise
    
    def _initialize_transform(self) -> None:
        """Initialize image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for feature extraction.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        try:
            if isinstance(image, str):
                # Load image from file path
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                # Use PIL Image directly
                pil_image = image.convert('RGB')
            else:
                raise ValueError("Image must be a file path or PIL Image object")
            
            # Apply transforms
            tensor = self.transform(pil_image)
            return tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def extract_features(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract features from an image using EfficientNet-B0.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            # Normalize features (L2 normalization)
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def extract_features_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Array of feature vectors
        """
        try:
            batch_features = []
            
            for image in images:
                features = self.extract_features(image)
                batch_features.append(features)
            
            return np.array(batch_features)
            
        except Exception as e:
            logger.error(f"Error extracting batch features: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'feature_dim': self.feature_dim,
            'input_size': (224, 224),
            'pretrained': True
        }

# Global instance for reuse
_image_processor = None

def get_image_processor() -> ImageProcessor:
    """Get or create a global image processor instance."""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor

@st.cache_resource
def load_image_processor() -> ImageProcessor:
    """Load image processor with Streamlit caching."""
    return ImageProcessor()

def extract_features_from_path(image_path: str) -> np.ndarray:
    """
    Convenience function to extract features from an image path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Feature vector as numpy array
    """
    processor = get_image_processor()
    return processor.extract_features(image_path)