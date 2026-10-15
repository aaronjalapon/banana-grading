# Banana Grading System

A Streamlit computer-vision prototype that detects bananas in RGB images and classifies maturity, quality, size, and view.

**Author:** Aaron Jalapon Arquillano  
[Portfolio case study](https://www.aaronarquillano.tech/projects/banana-grading-system)

![Recorded banana grading interface](https://raw.githubusercontent.com/aaronjalapon/portfolio/main/public/banana-grading.png)

The screenshot records an earlier interface state; its detector label says v5. The published inference source expects the v16 detector checkpoint described below.

## What I built

- A Faster R-CNN ResNet50-FPN-v2 detector with a banana/background prediction head.
- VGG16-based classifiers for four attributes, with optional ensemble inference and test-time augmentation.
- An image upload interface with detection thresholds, crop padding, bounding boxes, and per-attribute predictions.
- Training utility and model-module tests, plus saved evaluation reports from the local academic project.

## Pipeline

```text
RGB image → Faster R-CNN boxes → padded banana crops
          → VGG16 feature extraction → per-attribute classifier heads
          → maturity / quality / size / view labels and scores
```

| Attribute | Classes |
| --- | --- |
| Maturity | Green, Turning Yellow, Yellow, Overripe |
| Quality | Class 1, Class 2, Class 3 |
| Size | Large, Medium, Small |
| View | Front, Back |

## Release scope

This is an academic prototype and an inference-source release. It includes the Streamlit app, model wrappers, existing tests, and historical evaluation reports. Training images, model checkpoints, training notebooks, and manuscript files are not bundled. A fresh clone supports source review and the independent tests; image inference requires compatible checkpoints.

## Run locally

Use Python 3.11 with a compatible PyTorch/torchvision pair. CUDA is optional; the loader also supports MPS and CPU.

```bash
git clone https://github.com/aaronjalapon/banana-grading.git
cd banana-grading
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place trusted checkpoints from the matching v16 training run in saved_models/:

```text
saved_models/
├── best_detector_v16.pth
├── best_maturity_v16.pth
├── best_quality_v16.pth
├── best_size_v16.pth
└── best_view_v16.pth
```

Optional ensemble checkpoints use ensemble_maturity_v16.pth, ensemble_quality_v16.pth, ensemble_size_v16.pth, and ensemble_view_v16.pth. The classifier also initializes an ImageNet-pretrained VGG16 backbone, which torchvision may download on first use. Load only trusted checkpoints; classifier loading uses PyTorch serialization.

```bash
streamlit run app.py
```

Upload an RGB image, adjust the detection threshold and crop padding, then inspect the bounding boxes and classification details. The app reports missing checkpoints instead of producing predictions when required files are unavailable.

## Evaluation evidence

[evaluation/](evaluation/) contains saved artifacts from a prior local experiment, copied without changing their reported values:

- [Detection metrics](evaluation/detection_metrics.json): reports a 433/54/54 train/validation/test image split and detector AP of approximately 0.9675.
- [Classification reports](evaluation/all_classification_reports.json): results for the four attributes across the recorded configurations.
- [Per-class accuracy](evaluation/per_class_accuracy.csv): class supports and accuracy for baseline, optimized, single-model TTA, and ensemble TTA.

These are historical, dataset-specific results, not a fresh reproduction or a guarantee on new images. The detector report's scheduler label and recorded learning-rate sequence need reconciliation before claiming full training reproducibility. The source release omits the dataset and checkpoints needed to reproduce model-quality results.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The existing suite checks focal loss, feature-space augmentation, learning-rate scheduling, transforms, head architecture, and model-module interfaces. Some tests re-declare notebook utilities rather than importing inference code. Dataset-dependent tests skip when annotation CSVs are absent. The full-forward-pass test needs torchvision's pretrained VGG16 weights and can download them if uncached.

For the checks that do not need pretrained weights:

```bash
python -m pytest tests -q -k 'not full_forward_pass'
```

### Source-release verification — 15 October 2026

The published source package was checked with Python 3.11 using the command above: **41 passed, 3 skipped, 1 deselected**. The skipped tests require dataset annotation CSVs; the deselected test requires a pretrained VGG16 forward pass. This check does not reproduce the historical model-quality metrics.

## Source map

| Path | Purpose |
| --- | --- |
| app.py | Streamlit image upload, visualization, and inference workflow |
| model/detector.py | Faster R-CNN checkpoint loading and prediction |
| model/classifier_v16.py | VGG16 attribute heads, ensembles, and augmentation |
| model/transforms.py | Detector preprocessing |
| model/classifier.py | Legacy classifier helpers retained for package imports |
| tests/ | Existing training utility and model-module tests |
| evaluation/ | Historical JSON/CSV results |
