"""Streamlit app for banana grading detection and multi-attribute classification.

Uses a two-stage pipeline:
- Detection: Faster R-CNN detector (v16 checkpoint)
- Classification: v16 VGG16 per-attribute classifiers from saved_models/
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Banana Grading - Detection & Classification",
    page_icon="🍌",
    layout="wide"
)

# Font Awesome CSS
FONT_AWESOME_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
.fa-icon { font-size: 1em; margin-right: 0.3em; }
.fa-header { font-size: 1.2em; margin-right: 0.4em; }
</style>
"""

def fa_icon(icon_class: str, size: str = "") -> str:
    """Return Font Awesome icon HTML."""
    size_class = f"fa-{size}" if size else ""
    return f'<i class="{icon_class} {size_class}"></i>'

def fa_text(icon_class: str, text: str) -> str:
    """Return Font Awesome icon with text as HTML."""
    return f'<i class="{icon_class} fa-icon"></i>{text}'

# Constants
MODEL_DIR = Path(__file__).parent
V16_MODEL_DIR = MODEL_DIR / "saved_models"
DETECTOR_PATH_V16 = V16_MODEL_DIR / "best_detector_v16.pth"

# Main box color
BOX_COLOR = (255, 200, 0)  # Yellow


@st.cache_resource
def load_models_v16():
    """Load v5 detector + v16 VGG16 per-attribute classifiers (cached)."""
    from model import BananaDetector
    from model.classifier_v16 import load_v16_classifier, ATTRIBUTE_CLASS_NAMES, ATTR_FILE_KEY

    if not DETECTOR_PATH_V16.exists():
        st.error(f"Detector model not found: {DETECTOR_PATH_V16}")
        st.stop()

    if not V16_MODEL_DIR.exists():
        st.error(f"v16 model directory not found: {V16_MODEL_DIR}")
        st.stop()

    missing_files = []
    for attr_name in ATTRIBUTE_CLASS_NAMES:
        attr_key = ATTR_FILE_KEY[attr_name]
        single_ckpt = V16_MODEL_DIR / f"best_{attr_key}_v16.pth"
        if not single_ckpt.exists():
            missing_files.append(str(single_ckpt))

    if missing_files:
        st.error("Missing required v16 checkpoints:\n" + "\n".join(missing_files))
        st.stop()

    detector = BananaDetector(str(DETECTOR_PATH_V16))
    classifier = load_v16_classifier(
        model_dir=str(V16_MODEL_DIR),
        device=detector.device,
        load_ensemble=True
    )

    return detector, classifier


def draw_detections_with_attributes(
    image: Image.Image,
    predictions: dict,
    classifications: list
) -> Image.Image:
    """Draw bounding boxes with attribute labels on the image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to use a nicer font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
    
    boxes = predictions['boxes']
    scores = predictions['scores']
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        
        # Get classification results for this detection
        if i < len(classifications):
            attrs = classifications[i]
            
            # Build label lines
            lines = [f"Banana #{i+1} ({score:.0%})"]
            for attr_name in ['maturity', 'quality', 'size', 'view']:
                if attr_name in attrs:
                    label = attrs[attr_name]['label']
                    conf = attrs[attr_name]['confidence']
                    lines.append(f"{attr_name.capitalize()}: {label} ({conf:.0%})")
            
            # Calculate total height needed
            line_height = 16
            total_height = len(lines) * line_height + 8
            
            # Calculate max width
            max_width = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=small_font)
                max_width = max(max_width, bbox[2] - bbox[0])
            max_width += 12
            
            # Draw label background (above the box)
            label_y = max(0, y1 - total_height)
            draw.rectangle(
                [x1, label_y, x1 + max_width, label_y + total_height],
                fill=(0, 0, 0, 200)
            )
            
            # Draw each line
            current_y = label_y + 4
            for j, line in enumerate(lines):
                color = (255, 255, 255) if j == 0 else (220, 220, 220)
                draw.text((x1 + 6, current_y), line, fill=color, font=small_font)
                current_y += line_height
        else:
            # Fallback if no classification
            label = f"Banana: {score:.0%}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            draw.rectangle(
                [x1, y1 - text_h - 8, x1 + text_w + 8, y1],
                fill=BOX_COLOR
            )
            draw.text((x1 + 4, y1 - text_h - 4), label, fill=(0, 0, 0), font=font)
    
    return img_draw


def display_classification_details(classifications: list, predictions: dict):
    """Display detailed classification results in sidebar/main area."""
    if not classifications:
        st.info("No classifications to display.")
        return
    
    for i, (attrs, score) in enumerate(zip(classifications, predictions['scores'])):
        with st.expander(f"Banana #{i+1} (Detection: {score:.1%})", expanded=(i == 0), icon=":material/nutrition:"):
            # Create columns for each attribute
            cols = st.columns(4)
            
            for j, attr_name in enumerate(['maturity', 'quality', 'size', 'view']):
                if attr_name in attrs:
                    with cols[j]:
                        pred = attrs[attr_name]
                        st.markdown(f"**{attr_name.capitalize()}**")
                        st.markdown(f"### {pred['label']}")
                        st.progress(pred['confidence'], text=f"{pred['confidence']:.1%}")
                        
                        # Show all probabilities
                        st.markdown("*All probabilities:*")
                        for label, prob in sorted(
                            pred['all_probs'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            if prob > 0.5:
                                bar_icon = '<i class="fa-solid fa-circle" style="color: #28a745;"></i>'
                            elif prob > 0.2:
                                bar_icon = '<i class="fa-solid fa-circle" style="color: #ffc107;"></i>'
                            else:
                                bar_icon = '<i class="fa-solid fa-circle" style="color: #dfe4e8;"></i>'
                            st.markdown(f"{bar_icon} {label}: {prob:.1%}", unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    # Inject Font Awesome CSS
    st.markdown(FONT_AWESOME_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown(f'<h1> Banana Grading System</h1>', unsafe_allow_html=True)
    st.markdown("""
    Upload an image to **detect** bananas and **classify** their attributes:
    - **Maturity**: Green, Turning Yellow, Yellow, Overripe
    - **Quality**: Class 1, Class 2, Class 3
    - **Size**: Large, Medium, Small
    - **View**: Front, Back
    """)
    
    # Sidebar controls
    st.sidebar.markdown(f'<h3>{fa_icon("fa-solid fa-gear")} Settings</h3>', unsafe_allow_html=True)
    
    use_ensemble = st.sidebar.checkbox(
        "Use Ensemble (Top-5)", value=True,
        help="Average predictions from the saved top-5 ensemble heads per attribute"
    )
    use_tta = st.sidebar.checkbox(
        "Use TTA (16 views)", value=False,
        help="Test-time augmentation: 1 clean pass + 15 augmented passes (slower but more robust)"
    )
    
    score_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for banana detection"
    )
    
    padding_percent = st.sidebar.slider(
        "Crop Padding",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Padding around detected bananas for classification (15% recommended)"
    )
    
    # Load models
    classifier_v16 = None
    with st.spinner("Loading models..."):
        detector, classifier_v16 = load_models_v16()
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'### {fa_icon("fa-solid fa-chart-bar")} Model Info', unsafe_allow_html=True)
    st.sidebar.markdown(f"**Device:** `{detector.get_device_info()}`")
    st.sidebar.markdown(f"**Detector:** Faster R-CNN (v5)")
    st.sidebar.markdown(f"**Classifier:** VGG16 Per-Attribute (v16)")
    st.sidebar.markdown(f"**Ensemble:** {'Yes (top-5)' if use_ensemble else 'No'}")
    st.sidebar.markdown(f"**TTA:** {'Yes (16 views)' if use_tta else 'No'}")
    
    # Attribute info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'### {fa_icon("fa-solid fa-tags")} Attributes', unsafe_allow_html=True)
    from model.classifier_v16 import ATTRIBUTE_CLASS_NAMES
    for attr, labels in ATTRIBUTE_CLASS_NAMES.items():
        st.sidebar.markdown(f"**{attr}:** {', '.join(labels)}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload a banana image for detection and classification"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<h3>{fa_icon("fa-solid fa-camera")} Original Image</h3>', unsafe_allow_html=True)
            st.image(image)
        
        # Run detection
        with st.spinner("Detecting bananas..."):
            predictions = detector.predict(image, score_threshold=score_threshold)
        
        # Run classification on detected regions
        classifications = []
        if predictions['num_detections'] > 0:
            with st.spinner("Classifying attributes..."):
                from model.classifier_v16 import crop_with_padding
                
                # Crop detected regions with padding
                cropped_images = []
                for box in predictions['boxes']:
                    cropped = crop_with_padding(image, tuple(box), padding_percent)
                    cropped_images.append(cropped)

                if classifier_v16 is not None:
                    raw_results = classifier_v16.classify_batch(
                        cropped_images,
                        use_ensemble=use_ensemble,
                        use_tta=use_tta
                    )
                    attr_key_map = {
                        "Maturity Classification": "maturity",
                        "Quality Classification": "quality",
                        "Size Classification": "size",
                        "View Classification": "view",
                    }
                    for raw in raw_results:
                        converted = {}
                        for full_name, short_name in attr_key_map.items():
                            if full_name in raw:
                                converted[short_name] = raw[full_name]
                        classifications.append(converted)
        
        with col2:
            st.markdown(f'<h3>{fa_icon("fa-solid fa-magnifying-glass")} Detection & Classification</h3>', unsafe_allow_html=True)
            if predictions['num_detections'] > 0:
                result_image = draw_detections_with_attributes(
                    image, predictions, classifications
                )
                st.image(result_image)
            else:
                st.image(image)
                st.warning("No bananas detected above the confidence threshold.")
        
        # Detection & Classification details
        st.markdown("---")
        st.markdown(f'<h3>{fa_icon("fa-solid fa-chart-pie")} Results Summary</h3>', unsafe_allow_html=True)
        
        if predictions['num_detections'] > 0:
            # Summary metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Bananas Detected", predictions['num_detections'])
            with metric_cols[1]:
                st.metric("Avg Detection Conf.", f"{np.mean(predictions['scores']):.1%}")
            
            # Get most common maturity and quality
            if classifications:
                maturities = [c['maturity']['label'] for c in classifications if 'maturity' in c]
                qualities = [c['quality']['label'] for c in classifications if 'quality' in c]
                
                with metric_cols[2]:
                    if maturities:
                        most_common_maturity = max(set(maturities), key=maturities.count)
                        st.metric("Most Common Maturity", most_common_maturity)
                
                with metric_cols[3]:
                    if qualities:
                        most_common_quality = max(set(qualities), key=qualities.count)
                        st.metric("Most Common Quality", most_common_quality)
            
            # Detailed results
            st.markdown("---")
            st.markdown(f'<h3>{fa_icon("fa-solid fa-microscope")} Detailed Classification Results</h3>', unsafe_allow_html=True)
            display_classification_details(classifications, predictions)
            
            # Show cropped images
            if classifications:
                st.markdown("---")
                st.markdown(f'<h3>{fa_icon("fa-solid fa-images")} Detected Banana Regions</h3>', unsafe_allow_html=True)
                crop_cols = st.columns(min(len(classifications), 4))
                
                for i, (box, attrs) in enumerate(zip(predictions['boxes'], classifications)):
                    from model.classifier_v16 import crop_with_padding
                    cropped = crop_with_padding(image, tuple(box), padding_percent)
                    
                    col_idx = i % len(crop_cols)
                    with crop_cols[col_idx]:
                        st.image(cropped, caption=f"Banana #{i+1}")
                        maturity = attrs.get('maturity', {}).get('label', 'N/A')
                        quality = attrs.get('quality', {}).get('label', 'N/A')
                        size = attrs.get('size', {}).get('label', 'N/A')
                        st.caption(
                            f"**{maturity}** • "
                            f"**{quality}** • "
                            f"**{size}**"
                        )
        else:
            st.info("No detections to display. Try lowering the confidence threshold in the sidebar.")
    
    else:
        # Show instructions when no image uploaded
        st.markdown(f'<div class="stInfo">{fa_icon("fa-solid fa-hand-pointer")} Upload an image to get started!</div>', unsafe_allow_html=True)
        st.info("Select an image file using the uploader above.")
        
        # Sample usage
        with st.expander("How to use", icon=":material/info:"):
            st.markdown("""
            1. **Upload an image** using the file uploader above
            2. **Adjust settings** in the sidebar if needed:
               - *Detection Confidence*: Lower to detect more bananas
               - *Crop Padding*: Adjust context around detected bananas
                    - *Use Ensemble*: Average top-5 saved heads per attribute
                    - *Use TTA*: Run 16-view inference (1 clean + 15 augmented)
            3. **View detection results** with bounding boxes
            4. **Check classification details** for each detected banana:
               - Maturity stage
               - Quality class
               - Size category
               - View angle
            
            **Supported formats:** JPG, JPEG, PNG, BMP, WebP
            """)
        
        # Model architecture info
        with st.expander("Model Architecture", icon=":material/psychology:"):
            st.markdown("""
            This system uses a two-stage pipeline:
            
            **1. Detection (Faster R-CNN)**
            - Backbone: ResNet50-FPN-v2
            - Task: Localize bananas in the image
            - Output: Bounding boxes with confidence scores
            
            **2. Classification v16 (Per-Attribute VGG16)**
            - Backbone: VGG16 features (frozen, pretrained on ImageNet)
            - AdaptiveAvgPool(7x7) -> 25088-d feature vectors
            - Per-attribute FC heads with paper-exact hidden layer sizes
            - Optional top-5 ensemble averaging per attribute
            - Optional TTA with 16 total views (1 clean + 15 augmented)
            - Attributes: Maturity, Quality, Size, View
            """)


if __name__ == "__main__":
    main()
