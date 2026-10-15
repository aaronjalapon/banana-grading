"""Banana detector model loading and inference."""
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.tv_tensors import Image as TVImage

from .transforms import preprocess_image


class BananaDetector:
    """Banana detection model wrapper for inference."""
    
    # Class labels (background + banana)
    CLASSES = ['background', 'banana']
    NUM_CLASSES = 2
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the banana detector.
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _get_device(self, device: str = None) -> torch.device:
        """Auto-detect or use specified device."""
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_model(self, model_path: str):
        """Load the Faster R-CNN model with custom head."""
        # Match the v16 training notebook; Faster R-CNN handles resize internally.
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            min_size=600,
            max_size=800,
        )
        
        # Modify classification head for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.NUM_CLASSES)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        
        return model
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        img_array = np.array(pil_image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return TVImage(img_tensor)
    
    @torch.no_grad()
    def predict(self, image: Image.Image, score_threshold: float = 0.5) -> dict:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image in RGB format
            score_threshold: Minimum confidence score to include detection
            
        Returns:
            Dictionary containing:
                - boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] coordinates
                - scores: numpy array of shape (N,) with confidence scores
                - labels: numpy array of shape (N,) with class indices
                - class_names: list of class names for each detection
                - original_size: tuple (width, height) of input image
        """
        # Store original size
        orig_w, orig_h = image.size
        
        # Convert and preprocess
        img_tensor = self._pil_to_tensor(image)
        img_preprocessed = preprocess_image(img_tensor)
        
        # Run inference
        outputs = self.model([img_preprocessed.to(self.device)])[0]
        
        # Filter by score threshold
        mask = outputs['scores'].cpu() > score_threshold
        
        boxes = outputs['boxes'][mask].cpu().numpy()
        scores = outputs['scores'][mask].cpu().numpy()
        labels = outputs['labels'][mask].cpu().numpy()
        
        # Get class names
        class_names = [self.CLASSES[label] for label in labels]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'original_size': (orig_w, orig_h),
            'num_detections': len(boxes)
        }
    
    def get_device_info(self) -> str:
        """Return device information string."""
        return str(self.device)
