"""Multi-task banana classifier for attribute classification (v6 architecture)."""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2 as T
from PIL import Image


class MultiTaskBananaClassifier(nn.Module):
    """
    Multi-task classifier with ResNet50 backbone for banana attribute classification.
    
    Classifies 4 attributes:
        - maturity: Green, Overripe, Turning Yellow, Yellow
        - quality: Class 1, Class 2, Class 3
        - size: Large, Medium, Small
        - view: back, front
    """
    
    def __init__(self, attribute_info: dict, dropout: float = 0.3):
        """
        Initialize the multi-task classifier.
        
        Args:
            attribute_info: Dictionary containing attribute configurations with
                           'num_classes' for each attribute
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # ResNet50 Backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = backbone.fc.in_features  # 2048
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific classification heads
        self.heads = nn.ModuleDict()
        for attr, info in attribute_info.items():
            self.heads[attr] = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, info['num_classes'])
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary mapping attribute names to logit tensors
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        shared = self.shared_fc(features)
        
        outputs = {}
        for attr, head in self.heads.items():
            outputs[attr] = head(shared)
        
        return outputs


def get_classifier_transforms():
    """
    Get inference transforms for the classifier.
    
    Returns:
        Composed transforms for 224x224 ImageNet normalization
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_classifier(model_path: str, device: torch.device = None):
    """
    Load the multi-task classifier from a checkpoint.
    
    Args:
        model_path: Path to the checkpoint file (.pth)
        device: Device to load the model on
        
    Returns:
        Tuple of (model, attribute_info)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    attribute_info = checkpoint['attribute_info']
    
    model = MultiTaskBananaClassifier(attribute_info)
    
    # Handle DataParallel wrapped models (keys start with 'module.')
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, attribute_info


def crop_with_padding(image: Image.Image, box: tuple, padding_percent: float = 0.15) -> Image.Image:
    """
    Crop a region from an image with padding.
    
    Args:
        image: PIL Image to crop from
        box: Tuple of (x1, y1, x2, y2) bounding box coordinates
        padding_percent: Percentage of box dimensions to add as padding (0.15 = 15%)
        
    Returns:
        Cropped PIL Image with padding
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Calculate padding
    pad_w = width * padding_percent
    pad_h = height * padding_percent
    
    # Apply padding while staying within image bounds
    img_w, img_h = image.size
    x1_padded = max(0, int(x1 - pad_w))
    y1_padded = max(0, int(y1 - pad_h))
    x2_padded = min(img_w, int(x2 + pad_w))
    y2_padded = min(img_h, int(y2 + pad_h))
    
    return image.crop((x1_padded, y1_padded, x2_padded, y2_padded))


@torch.no_grad()
def classify_attributes(
    model: MultiTaskBananaClassifier,
    image: Image.Image,
    attribute_info: dict,
    device: torch.device
) -> dict:
    """
    Classify attributes for a single cropped banana image.
    
    Args:
        model: Loaded MultiTaskBananaClassifier
        image: PIL Image (cropped banana region)
        attribute_info: Dictionary with attribute configurations
        device: Device to run inference on
        
    Returns:
        Dictionary with predictions for each attribute
    """
    transform = get_classifier_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    outputs = model(img_tensor)
    
    results = {}
    for attr, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = attribute_info[attr]['labels'][pred_idx]
        confidence = probs[pred_idx].item()
        
        results[attr] = {
            'label': pred_label,
            'confidence': confidence,
            'all_probs': {
                label: probs[i].item()
                for i, label in enumerate(attribute_info[attr]['labels'])
            }
        }
    
    return results


@torch.no_grad()
def classify_batch(
    model: MultiTaskBananaClassifier,
    images: list,
    attribute_info: dict,
    device: torch.device
) -> list:
    """
    Classify attributes for a batch of cropped banana images.
    
    Args:
        model: Loaded MultiTaskBananaClassifier
        images: List of PIL Images (cropped banana regions)
        attribute_info: Dictionary with attribute configurations
        device: Device to run inference on
        
    Returns:
        List of dictionaries with predictions for each image
    """
    if not images:
        return []
    
    transform = get_classifier_transforms()
    batch_tensors = torch.stack([transform(img) for img in images]).to(device)
    
    outputs = model(batch_tensors)
    
    results = []
    batch_size = len(images)
    
    for i in range(batch_size):
        img_results = {}
        for attr, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)[i]
            pred_idx = torch.argmax(probs).item()
            pred_label = attribute_info[attr]['labels'][pred_idx]
            confidence = probs[pred_idx].item()
            
            img_results[attr] = {
                'label': pred_label,
                'confidence': confidence,
                'all_probs': {
                    label: probs[j].item()
                    for j, label in enumerate(attribute_info[attr]['labels'])
                }
            }
        results.append(img_results)
    
    return results
