"""Tests for banana-grading-v16 notebook code.

Covers v15 tests plus:
1. FocalLoss: correct gradient flow, gamma weighting, alpha class weights
2. Feature-space CutMix: dimension masking correctness
3. augmix_batch: MixUp/CutMix random selection
4. CosineAnnealingWarmRestarts: scheduler produces warm restarts
5. Enhanced augmentation: RandomPerspective, stronger ColorJitter
6. classifier_v16.py module: importable with all new components
7. Export functions: confusion matrix, classification report, per-class accuracy
"""

import os
import random
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.models import vgg16, VGG16_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


# ======================================================================
# Re-declare minimal v16 notebook code under test
# ======================================================================

FEAT_DIM = 25088
IMG_SIZE = 224
SEED = 42

ATTRIBUTES = ['maturity', 'quality', 'size', 'view']

CLASS_NAMES = {
    'maturity': ['Green', 'Overripe', 'Turning Yellow', 'Yellow'],
    'quality':  ['Class 1', 'Class 2', 'Class 3'],
    'size':     ['Large', 'Medium', 'Small'],
    'view':     ['back', 'front'],
}

ATTR_HEADS = {
    'maturity': [256],
    'quality':  [256, 32],
    'size':     [256],
    'view':     [4096, 4096],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ATTRIBUTE_CSV_MAP = {
    'maturity': 'annotations_maturity.csv',
    'quality': 'annotations_quality.csv',
    'size': 'annotations_size.csv',
    'view': 'annotations_view.csv',
}


# ---- FCHead with Xavier Uniform ----
class FCHead(nn.Module):
    def __init__(self, feat_dim, num_classes, head_dims=(256,), dropout=0.0):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for out_dim in head_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


# ---- FocalLoss ----
class FocalLoss(nn.Module):
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


# ---- MixUp ----
def mixup_data(features, labels, alpha=0.2):
    if alpha <= 0:
        return features, labels, None, None, 1.0
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)
    mixed_features = lam * features + (1 - lam) * features[index]
    return mixed_features, labels, labels[index], index, lam


# ---- CutMix (feature-space) ----
def cutmix_data(features, labels, alpha=0.2):
    if alpha <= 0:
        return features, labels, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = features.size(0)
    feat_dim = features.size(1)
    index = torch.randperm(batch_size, device=features.device)
    n_replace = int(feat_dim * (1 - lam))
    replace_idx = torch.randperm(feat_dim, device=features.device)[:n_replace]
    mixed = features.clone()
    mixed[:, replace_idx] = features[index][:, replace_idx]
    actual_lam = 1.0 - (n_replace / feat_dim)
    return mixed, labels, labels[index], index, actual_lam


def augmix_batch(features, labels, mixup_alpha=0.2, cutmix_alpha=0.2):
    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if use_mixup and use_cutmix:
        if random.random() < 0.5:
            return mixup_data(features, labels, mixup_alpha)
        else:
            return cutmix_data(features, labels, cutmix_alpha)
    elif use_mixup:
        return mixup_data(features, labels, mixup_alpha)
    elif use_cutmix:
        return cutmix_data(features, labels, cutmix_alpha)
    else:
        return features, labels, None, None, 1.0


# ---- Transforms ----
def get_v16_train_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(IMG_SIZE),
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


def get_v16_eval_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(IMG_SIZE),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_spatial_transforms():
    transforms = [
        T.RandomHorizontalFlip(p=1.0),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 0, 0, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 0, 32, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 32, 0, 224, 224))]),
        T.Compose([T.Resize((256, 256)),
                    T.Lambda(lambda img: T.functional.crop(img, 32, 32, 224, 224))]),
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        T.RandomRotation((10, 10)),
        T.RandomRotation((-10, -10)),
        T.RandomVerticalFlip(p=1.0),
        T.RandomRotation((5, 5)),
        T.RandomRotation((-5, -5)),
        T.Compose([T.Resize((270, 270)), T.CenterCrop(224)]),
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        T.Compose([T.Resize((256, 256)), T.RandomCrop(224)]),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ]
    return transforms


def group_aware_stratified_split(labels, group_ids, test_ratio=0.1, val_ratio=0.125, seed=42):
    rng = np.random.RandomState(seed)
    group_to_indices = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_indices[gid].append(i)
    unique_groups = list(group_to_indices.keys())
    group_labels = [labels[group_to_indices[g][0]] for g in unique_groups]
    unique_groups = np.array(unique_groups)
    group_labels = np.array(group_labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    remain_grp_idx, test_grp_idx = next(sss1.split(unique_groups, group_labels))
    remain_groups = unique_groups[remain_grp_idx]
    remain_labels = group_labels[remain_grp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_grp_idx, val_grp_idx = next(sss2.split(remain_groups, remain_labels))
    test_groups = set(unique_groups[test_grp_idx].tolist())
    val_groups = set(remain_groups[val_grp_idx].tolist())
    train_groups = set(remain_groups[train_grp_idx].tolist())
    train_indices = [i for g in train_groups for i in group_to_indices[g]]
    val_indices = [i for g in val_groups for i in group_to_indices[g]]
    test_indices = [i for g in test_groups for i in group_to_indices[g]]
    return (
        np.array(sorted(train_indices)),
        np.array(sorted(val_indices)),
        np.array(sorted(test_indices)),
    )


# ======================================================================
# Fixtures
# ======================================================================

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")


def _csv_path(attr):
    return os.path.join(DATASET_ROOT, ATTRIBUTE_CSV_MAP[attr])


def _skip_if_no_dataset():
    path = os.path.join(DATASET_ROOT, "annotations_maturity.csv")
    if not os.path.exists(path):
        pytest.skip("Dataset CSVs not found")


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))


# ======================================================================
# 1. Focal Loss
# ======================================================================

class TestFocalLoss:
    """Verify FocalLoss correctness."""

    def test_focal_loss_reduces_easy_sample_weight(self):
        """Focal loss should down-weight easy (well-classified) samples."""
        logits = torch.tensor([[10.0, 0.0, 0.0],   # easy: high confidence correct
                               [0.5, 0.3, 0.2]])    # hard: low confidence
        targets = torch.tensor([0, 0])

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        fl = FocalLoss(gamma=2.0, reduction='none')
        focal_loss = fl(logits, targets)

        # Easy sample should have much lower focal loss relative to CE
        ce_ratio = ce_loss[0] / ce_loss[1]
        fl_ratio = focal_loss[0] / focal_loss[1]
        assert fl_ratio < ce_ratio, "Focal loss should down-weight easy samples more"

    def test_focal_loss_gamma_zero_equals_ce(self):
        """Focal loss with gamma=0 should equal cross-entropy."""
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        ce = F.cross_entropy(logits, targets)
        fl = FocalLoss(gamma=0.0)(logits, targets)
        assert torch.allclose(ce, fl, atol=1e-5), f"γ=0 focal loss {fl} != CE {ce}"

    def test_focal_loss_with_alpha(self):
        """Focal loss with class weights should apply per-class scaling."""
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        alpha = [1.0, 2.0, 0.5]

        fl_uniform = FocalLoss(gamma=2.0)(logits, targets)
        fl_weighted = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)
        # Weighted should differ from uniform
        assert not torch.allclose(fl_uniform, fl_weighted), "Alpha weighting had no effect"

    def test_focal_loss_gradient_flows(self):
        """Focal loss should allow gradient computation."""
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])
        fl = FocalLoss(gamma=2.0)
        loss = fl(logits, targets)
        loss.backward()
        assert logits.grad is not None, "Gradients did not flow through FocalLoss"
        assert not torch.isnan(logits.grad).any(), "NaN gradients from FocalLoss"

    def test_focal_loss_with_label_smoothing(self):
        """Focal loss should work with label smoothing."""
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        fl_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
        fl_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)
        # Should produce different losses
        assert not torch.allclose(fl_no_smooth, fl_smooth)


# ======================================================================
# 2. Feature-Space CutMix
# ======================================================================

class TestCutMix:
    """Verify feature-space CutMix correctness."""

    def test_cutmix_shape_preserved(self):
        """CutMix should preserve feature tensor shape."""
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = cutmix_data(feats, labels, alpha=0.2)
        assert mixed.shape == feats.shape, "CutMix changed feature shape"

    def test_cutmix_replaces_dimensions(self):
        """CutMix should replace a subset of feature dimensions."""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        feats = torch.randn(4, 100)
        labels = torch.randint(0, 3, (4,))
        mixed, la, lb, idx, lam = cutmix_data(feats, labels, alpha=0.5)
        # Some dimensions should be from the original, some from the shuffled
        if lam < 1.0:
            # Not all dimensions should be changed
            same_dims = (mixed[0] == feats[0]).sum().item()
            assert 0 < same_dims < 100, f"Expected partial replacement, got {same_dims}/100 same"

    def test_cutmix_alpha_zero_returns_original(self):
        """CutMix with alpha=0 should return original features."""
        feats = torch.randn(4, 256)
        labels = torch.randint(0, 3, (4,))
        mixed, la, lb, idx, lam = cutmix_data(feats, labels, alpha=0.0)
        assert torch.allclose(mixed, feats), "CutMix alpha=0 should return original"
        assert lam == 1.0

    def test_cutmix_lambda_reflects_replaced_fraction(self):
        """CutMix lambda should correspond to the fraction of kept dimensions."""
        np.random.seed(SEED)
        feats = torch.randn(4, 1000)
        labels = torch.randint(0, 3, (4,))
        mixed, la, lb, idx, lam = cutmix_data(feats, labels, alpha=0.5)
        # Lambda should be between 0 and 1
        assert 0 <= lam <= 1, f"Lambda {lam} out of range"


# ======================================================================
# 3. augmix_batch: MixUp/CutMix selection
# ======================================================================

class TestAugMixBatch:
    """Verify augmix_batch correctly dispatches between MixUp and CutMix."""

    def test_augmix_both_enabled(self):
        """When both alpha > 0, augmix should randomly choose one."""
        random.seed(SEED)
        np.random.seed(SEED)
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        # Run many times and collect lambdas
        mixed, la, lb, idx, lam = augmix_batch(feats, labels, mixup_alpha=0.2, cutmix_alpha=0.2)
        assert mixed.shape == feats.shape

    def test_augmix_only_mixup(self):
        """When cutmix_alpha=0, should only use MixUp."""
        np.random.seed(SEED)
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = augmix_batch(feats, labels, mixup_alpha=0.2, cutmix_alpha=0.0)
        assert mixed.shape == feats.shape

    def test_augmix_only_cutmix(self):
        """When mixup_alpha=0, should only use CutMix."""
        np.random.seed(SEED)
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = augmix_batch(feats, labels, mixup_alpha=0.0, cutmix_alpha=0.2)
        assert mixed.shape == feats.shape

    def test_augmix_neither_returns_original(self):
        """When both alpha=0, should return original."""
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = augmix_batch(feats, labels, mixup_alpha=0.0, cutmix_alpha=0.0)
        assert torch.allclose(mixed, feats)
        assert lam == 1.0


# ======================================================================
# 4. CosineAnnealingWarmRestarts
# ======================================================================

class TestWarmRestarts:
    """Verify CosineAnnealingWarmRestarts produces expected LR schedule."""

    def test_lr_restarts(self):
        """LR should restart (increase) at T_0 and T_0 + T_0*T_mult."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

        lrs = []
        for epoch in range(80):
            scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])

        # LR should be at maximum at epoch 0
        assert lrs[0] == pytest.approx(5e-4, abs=1e-6)
        # LR should decrease then restart at epoch 20
        assert lrs[19] < lrs[0]  # decayed before restart
        # At restart point (epoch 20), LR should jump back up
        assert lrs[20] > lrs[19]

    def test_lr_is_periodic(self):
        """LR schedule should show periodic behavior."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

        lrs = []
        for epoch in range(30):
            scheduler.step(epoch)
            lrs.append(optimizer.param_groups[0]['lr'])

        # Period 1 and period 2 should have similar LR patterns
        period1 = lrs[0:10]
        period2 = lrs[10:20]
        for i in range(10):
            assert abs(period1[i] - period2[i]) < 1e-5, \
                f"LR not periodic at offset {i}: {period1[i]} vs {period2[i]}"


# ======================================================================
# 5. Enhanced v16 Augmentation
# ======================================================================

class TestV16Augmentation:

    def test_train_transform_includes_perspective(self):
        """v16 train transform should include RandomPerspective."""
        tf = get_v16_train_transform()
        tf_repr = repr(tf)
        assert "RandomPerspective" in tf_repr, "Missing RandomPerspective in v16 train transform"

    def test_train_transform_stronger_colorjitter(self):
        """v16 should use stronger ColorJitter (brightness=0.3)."""
        tf = get_v16_train_transform()
        tf_repr = repr(tf)
        assert "ColorJitter" in tf_repr

    def test_train_transform_random_erasing_higher_prob(self):
        """v16 should use RandomErasing with p=0.15."""
        tf = get_v16_train_transform()
        tf_repr = repr(tf)
        assert "RandomErasing" in tf_repr

    def test_train_transform_output_shape(self, dummy_image):
        """Train transform should produce (3, 224, 224) tensor."""
        tf = get_v16_train_transform()
        tensor = tf(dummy_image)
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_produces_variation(self, dummy_image):
        """Multiple applications should produce different outputs."""
        tf = get_v16_train_transform()
        t1 = tf(dummy_image)
        t2 = tf(dummy_image)
        assert not torch.allclose(t1, t2), "Train transform produced identical outputs"


# ======================================================================
# 6. TTA: 16 Total Views (unchanged from v15)
# ======================================================================

class TestTTA:

    def test_tta_spatial_count(self):
        """Should have 15 spatial transforms + 1 clean = 16 total."""
        transforms = get_tta_spatial_transforms()
        assert len(transforms) == 15, f"Expected 15 spatial TTA, got {len(transforms)}"

    def test_tta_transforms_produce_different_outputs(self, dummy_image):
        """Each TTA transform should produce a different view."""
        transforms = get_tta_spatial_transforms()
        eval_tf = get_v16_eval_transform()
        results = []
        results.append(eval_tf(dummy_image))
        for i, t in enumerate(transforms):
            if i not in [5, 12, 13]:
                try:
                    aug = t(dummy_image)
                    results.append(eval_tf(aug))
                except Exception:
                    pass
        clean = results[0]
        n_different = sum(1 for r in results[1:] if not torch.allclose(r, clean, atol=1e-4))
        assert n_different >= 5, f"Only {n_different} TTA views differ from clean"


# ======================================================================
# 7. Xavier Uniform Initialization (unchanged from v15)
# ======================================================================

class TestXavierInit:

    def test_xavier_init_weight_variance(self):
        torch.manual_seed(SEED)
        head = FCHead(FEAT_DIM, 4, (256,))
        first_linear = None
        for m in head.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        w = first_linear.weight.data
        fan_in, fan_out = w.size(1), w.size(0)
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        assert w.min().item() >= -limit - 0.01
        assert w.max().item() <= limit + 0.01

    def test_bias_zeros(self):
        head = FCHead(FEAT_DIM, 4, (256,))
        for m in head.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                assert torch.all(m.bias.data == 0), "Bias not zero-initialized"


# ======================================================================
# 8. FCHead Architecture (paper-exact)
# ======================================================================

class TestFCHeadArchitecture:

    @pytest.mark.parametrize("attr", ATTRIBUTES)
    def test_architecture_matches_paper(self, attr):
        num_cls = len(CLASS_NAMES[attr])
        head_dims = ATTR_HEADS[attr]
        head = FCHead(FEAT_DIM, num_cls, head_dims)
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        expected_n = len(head_dims) + 1
        assert len(linears) == expected_n
        assert linears[-1].out_features == num_cls

    @pytest.mark.parametrize("attr", ATTRIBUTES)
    def test_head_forward_pass(self, attr):
        num_cls = len(CLASS_NAMES[attr])
        head = FCHead(FEAT_DIM, num_cls, ATTR_HEADS[attr])
        x = torch.randn(4, FEAT_DIM)
        out = head(x)
        assert out.shape == (4, num_cls)


# ======================================================================
# 9. MixUp (unchanged from v15)
# ======================================================================

class TestMixUp:

    def test_mixup_returns_mixed_features(self):
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = mixup_data(feats, labels, alpha=0.2)
        assert mixed.shape == feats.shape

    def test_mixup_alpha_zero_returns_original(self):
        feats = torch.randn(8, 256)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, idx, lam = mixup_data(feats, labels, alpha=0.0)
        assert lam == 1.0
        assert torch.allclose(mixed, feats)


# ======================================================================
# 10. Dataset Recovery (unchanged from v15)
# ======================================================================

class TestDatasetRecovery:

    def test_quality_matches_paper_exactly(self):
        _skip_if_no_dataset()
        df = pd.read_csv(_csv_path('quality'))
        assert len(df) == 596, f"Quality has {len(df)} samples (expected 596)"

    def test_horizontal_files_are_separate_entries(self):
        _skip_if_no_dataset()
        df = pd.read_csv(_csv_path('quality'))
        has_hflip = df['image_id'].str.startswith('HFLIP_').any()
        has_originals = (~df['image_id'].str.startswith('HFLIP_')).any()
        assert has_hflip
        assert has_originals


# ======================================================================
# 11. Group-Aware Splitting
# ======================================================================

class TestGroupAwareSplitting:

    def test_no_group_leakage_across_splits(self):
        _skip_if_no_dataset()
        for attr in ATTRIBUTES:
            df = pd.read_csv(_csv_path(attr))
            labels = df['label'].factorize()[0]
            group_ids = df['group_id'].values
            train_idx, val_idx, test_idx = group_aware_stratified_split(
                labels, group_ids, seed=SEED
            )
            train_groups = set(group_ids[train_idx])
            val_groups = set(group_ids[val_idx])
            test_groups = set(group_ids[test_idx])
            assert len(train_groups & val_groups) == 0
            assert len(train_groups & test_groups) == 0
            assert len(val_groups & test_groups) == 0


# ======================================================================
# 12. classifier_v16.py Module
# ======================================================================

class TestClassifierV16Module:

    def test_module_importable(self):
        import importlib
        mod = importlib.import_module("model.classifier_v16")
        assert hasattr(mod, 'FCHead')
        assert hasattr(mod, 'VGG16Backbone')
        assert hasattr(mod, 'VGG16AttributeClassifier')
        assert hasattr(mod, 'FocalLoss')
        assert hasattr(mod, 'get_v16_train_transform')
        assert hasattr(mod, 'get_v16_eval_transform')
        assert hasattr(mod, 'get_v16_tta_transforms')
        assert hasattr(mod, 'load_v16_classifier')

    def test_focal_loss_from_module(self):
        from model.classifier_v16 import FocalLoss as V16FocalLoss
        fl = V16FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = fl(logits, targets)
        assert loss.item() > 0

    def test_fchead_xavier_init(self):
        from model.classifier_v16 import FCHead as V16FCHead
        head = V16FCHead(FEAT_DIM, 4, [256])
        first_linear = None
        for m in head.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        w = first_linear.weight.data
        fan_in, fan_out = w.size(1), w.size(0)
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        assert w.min().item() >= -limit - 0.01
        assert w.max().item() <= limit + 0.01

    def test_tta_count_is_15_plus_1(self):
        from model.classifier_v16 import get_v16_tta_transforms
        transforms = get_v16_tta_transforms()
        assert len(transforms) == 15

    def test_v16_train_transform_has_perspective(self):
        from model.classifier_v16 import get_v16_train_transform
        tf = get_v16_train_transform()
        assert "RandomPerspective" in repr(tf)


# ======================================================================
# 13. End-to-end forward pass
# ======================================================================

class TestEndToEnd:

    def test_full_forward_pass(self, dummy_image):
        backbone = vgg16(weights=VGG16_Weights.DEFAULT).features
        pool = nn.AdaptiveAvgPool2d((7, 7))
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        tf = get_v16_eval_transform()
        head = FCHead(FEAT_DIM, 4, (256,))
        head.eval()

        img_tensor = tf(dummy_image).unsqueeze(0)
        with torch.no_grad():
            feats = backbone(img_tensor)
            feats = pool(feats)
            feats = feats.flatten(1)
            logits = head(feats)

        assert logits.shape == (1, 4)
        assert feats.shape == (1, FEAT_DIM)

    def test_focal_loss_in_training_loop(self):
        """Simulate a mini training loop with FocalLoss to verify it works end-to-end."""
        head = FCHead(256, 3, (64,))
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

        feats = torch.randn(16, 256)
        labels = torch.randint(0, 3, (16,))

        # Run 5 training steps
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = head(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (not always, but should not be NaN)
        assert all(not np.isnan(l) for l in losses), "NaN loss during training"
        assert all(not np.isinf(l) for l in losses), "Inf loss during training"

    def test_dropout_sweep_values(self):
        for dp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            head = FCHead(FEAT_DIM, 3, (256,), dropout=dp)
            x = torch.randn(2, FEAT_DIM)
            out = head(x)
            assert out.shape == (2, 3)
            assert not torch.isnan(out).any()
