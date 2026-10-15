"""VGG16-based banana classifier for v16 deployment.

This module provides per-attribute VGG16 classifiers (frozen backbone + FC head)
trained in banana-grading-v16.ipynb.

Key changes from v15:
    - Focal Loss option (gamma=2.0) for better class imbalance handling
    - Feature-space CutMix augmentation alongside MixUp
    - CosineAnnealingWarmRestarts scheduler (T0=20, Tmult=2) for multiple restarts
    - Stronger augmentation: ColorJitter(0.3), RandomPerspective, more RandomErasing
    - Training history plots, confusion matrix PNG export, classification report CSV/JSON
    - ZIP bundle of all evaluation artifacts for easy download

Architecture (unchanged from paper):
    - Backbone: VGG16 features (frozen, ImageNet pretrained)
    - AdaptiveAvgPool2d(7,7) -> flatten -> 25088-d feature vector
    - Per-attribute FC head with paper-exact hidden layer sizes
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import v2 as T
from PIL import Image
from typing import Optional


# ---- Paper-exact class names (sorted alphabetically, same as training) ----
ATTRIBUTE_CLASS_NAMES = {
    "Maturity Classification": ["Green", "Overripe", "Turning Yellow", "Yellow"],
    "Quality Classification": ["Class 1", "Class 2", "Class 3"],
    "Size Classification": ["Large", "Medium", "Small"],
    "View Classification": ["back", "front"],
}

# Short keys used in checkpoint filenames (best_<key>_v16.pth)
ATTR_FILE_KEY = {
    "Maturity Classification": "maturity",
    "Quality Classification": "quality",
    "Size Classification": "size",
    "View Classification": "view",
}

# ---- Paper-exact hidden layer configurations ----
PAPER_HIDDEN_LAYERS = {
    "Maturity Classification": [256],
    "Quality Classification": [256, 32],
    "Size Classification": [256],
    "View Classification": [4096, 4096],
}

FEAT_DIM = 25088  # VGG16 features: 512 * 7 * 7

# ---- Torchvision ImageNet RGB normalization ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples.
        alpha: Per-class weights (list/tensor) or None for uniform weighting.
        label_smoothing: Optional label smoothing (applied before focal weighting).
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal_weight = focal_weight * at

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FCHead(nn.Module):
    """Fully-connected classification head with Xavier Uniform init (Keras default).

    Architecture matches the paper exactly per attribute.
    """

    def __init__(self, feat_dim: int, num_classes: int,
                 hidden_layers: list, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

        # Xavier Uniform init (matches Keras/Glorot default used in the paper)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VGG16Backbone(nn.Module):
    """Frozen VGG16 feature extractor with adaptive pooling."""

    def __init__(self):
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # [B, 25088]


class VGG16AttributeClassifier:
    """Per-attribute VGG16 classifier supporting single/ensemble inference + enhanced TTA.

    Usage:
        classifier = VGG16AttributeClassifier.from_saved_models(
            model_dir="saved_models_v16", device="cpu"
        )
        results = classifier.classify(pil_image, use_tta=True, use_ensemble=True)
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device
        self.backbone = VGG16Backbone().to(device).eval()
        self.heads = {}       # attr -> FCHead (single best model)
        self.ensemble = {}    # attr -> [FCHead, ...] (multi-seed or K-fold models)
        self.class_names = {}  # attr -> [class_name, ...]
        self.transform = get_v16_eval_transform()
        self.tta_transforms = get_v16_tta_transforms()

    @classmethod
    def from_saved_models(cls, model_dir: str = "saved_models_v16",
                          device: Optional[torch.device] = None,
                          load_ensemble: bool = True) -> "VGG16AttributeClassifier":
        """Load all per-attribute models from the saved directory."""
        obj = cls(device=device)

        for attr, class_names in ATTRIBUTE_CLASS_NAMES.items():
            attr_key = ATTR_FILE_KEY[attr]
            obj.class_names[attr] = class_names

            # Load single best model
            single_path = os.path.join(model_dir, f"best_{attr_key}_v16.pth")
            if os.path.exists(single_path):
                ckpt = torch.load(single_path, map_location=obj.device, weights_only=False)
                head = FCHead(
                    ckpt["feat_dim"], ckpt["num_classes"],
                    ckpt["hidden_layers"], ckpt.get("dropout", 0.0)
                )
                head.load_state_dict(ckpt["head_state_dict"])
                head.to(obj.device).eval()
                obj.heads[attr] = head

            # Load ensemble models (multi-seed top-5)
            if load_ensemble:
                ens_path = os.path.join(model_dir, f"ensemble_{attr_key}_v16.pth")
                if os.path.exists(ens_path):
                    ckpt = torch.load(ens_path, map_location=obj.device, weights_only=False)
                    fold_heads = []
                    for state in ckpt["fold_states"]:
                        h = FCHead(
                            ckpt["feat_dim"], ckpt["num_classes"],
                            ckpt["hidden_layers"], ckpt.get("dropout", 0.0)
                        )
                        h.load_state_dict(state)
                        h.to(obj.device).eval()
                        fold_heads.append(h)
                    obj.ensemble[attr] = fold_heads

        return obj

    @torch.no_grad()
    def extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Extract VGG16 features from a preprocessed image tensor."""
        return self.backbone(img_tensor.to(self.device))

    @torch.no_grad()
    def classify(self, image: Image.Image, use_ensemble: bool = False,
                 use_tta: bool = False) -> dict:
        """Classify all attributes for a single banana image.

        Args:
            image: PIL Image (cropped banana region)
            use_ensemble: Use multi-seed ensemble averaging
            use_tta: Use enhanced TTA (16 spatial views)

        Returns:
            Dict mapping attribute name -> {label, confidence, all_probs}
        """
        results = {}

        if use_tta:
            features_list = []
            # Clean pass
            clean_tensor = self.transform(image).unsqueeze(0)
            features_list.append(self.extract_features(clean_tensor))
            # Augmented passes
            for tfm in self.tta_transforms:
                aug_img = tfm(image)
                aug_tensor = self.transform(aug_img).unsqueeze(0)
                features_list.append(self.extract_features(aug_tensor))
        else:
            img_tensor = self.transform(image).unsqueeze(0)
            features_list = [self.extract_features(img_tensor)]

        for attr in ATTRIBUTE_CLASS_NAMES:
            if attr not in self.heads and attr not in self.ensemble:
                continue

            class_names = self.class_names[attr]
            all_logits = []

            heads_to_use = (self.ensemble.get(attr, [self.heads[attr]])
                            if use_ensemble and attr in self.ensemble
                            else [self.heads[attr]])

            for feat in features_list:
                for head in heads_to_use:
                    all_logits.append(head(feat))

            avg_logits = torch.stack(all_logits).mean(dim=0)  # [1, num_cls]
            probs = torch.softmax(avg_logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

            results[attr] = {
                "label": class_names[pred_idx],
                "confidence": probs[pred_idx].item(),
                "all_probs": {
                    name: probs[i].item()
                    for i, name in enumerate(class_names)
                },
            }

        return results

    @torch.no_grad()
    def classify_batch(self, images: list, use_ensemble: bool = False,
                       use_tta: bool = False) -> list:
        """Classify all attributes for a batch of images."""
        return [self.classify(img, use_ensemble=use_ensemble, use_tta=use_tta)
                for img in images]


def get_v16_eval_transform():
    """Eval transform: Resize 256 -> CenterCrop 224 -> RGB Normalize (unchanged from v15)."""
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_v16_train_transform():
    """v16 enhanced training augmentation.

    Changes from v15:
    - Stronger ColorJitter (0.3/0.3/0.3/0.08 vs 0.2/0.2/0.2/0.05)
    - RandomPerspective (distortion_scale=0.1, p=0.2)
    - Increased RandomErasing (p=0.15, scale up to 0.20)
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(20),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomPerspective(distortion_scale=0.1, p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.20)),
    ])


def get_v16_tta_transforms():
    """Enhanced TTA: 15 spatial transforms (+ 1 clean pass = 16 total views)."""
    transforms = [
        # 0: Horizontal flip
        T.RandomHorizontalFlip(p=1.0),
        # 1-4: Positional crops (corners of 256x256 -> 224x224)
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 0, 0, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 0, 32, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 32, 0, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 32, 32, 224, 224))]),
        # 5: Random crop
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        # 6-7: Slight rotations
        T.RandomRotation((10, 10)),
        T.RandomRotation((-10, -10)),
        # 8: Vertical flip
        T.RandomVerticalFlip(p=1.0),
        # 9-10: Finer rotations
        T.RandomRotation((5, 5)),
        T.RandomRotation((-5, -5)),
        # 11: Multi-scale crop (resize larger then center crop)
        T.Compose([T.Resize((270, 270)), T.CenterCrop(224)]),
        # 12-13: Additional random crops for diversity
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        # 14: Color-shifted view (subtle jitter that doesn't destroy info)
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ]
    return transforms


def load_v16_classifier(model_dir: str = "saved_models_v16",
                        device: Optional[torch.device] = None,
                        load_ensemble: bool = True) -> VGG16AttributeClassifier:
    """Convenience function to load the v16 classifier."""
    return VGG16AttributeClassifier.from_saved_models(
        model_dir=model_dir, device=device, load_ensemble=load_ensemble
    )


# ---- Compatibility with app.py crop_with_padding ----
def crop_with_padding(image: Image.Image, box: tuple,
                      padding_percent: float = 0.15) -> Image.Image:
    """Crop a region from an image with padding (same as v6 classifier)."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_w = width * padding_percent
    pad_h = height * padding_percent
    img_w, img_h = image.size
    x1_padded = max(0, int(x1 - pad_w))
    y1_padded = max(0, int(y1 - pad_h))
    x2_padded = min(img_w, int(x2 + pad_w))
    y2_padded = min(img_h, int(y2 + pad_h))
    return image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
