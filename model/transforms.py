"""Image preprocessing transforms for banana detection."""
import torch


def get_transforms():
    """
    Get inference transforms matching the Faster R-CNN training pipeline.
    
    Returns:
        Callable that converts uint8 image tensors to float tensors in [0, 1].
    """
    return preprocess_image


def preprocess_image(image_tensor):
    """
    Preprocess an image tensor for model inference.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) in uint8 format
        
    Returns:
        Preprocessed tensor ready for model input
    """
    return image_tensor.to(dtype=torch.float32) / 255.0
