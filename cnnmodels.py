from keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import io

CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Load models ONCE
detection_model = TFSMLayer(
    "BrainTumorDetectionModel/model",
    call_endpoint="serving_default"
)

classification_model = TFSMLayer(
    "BrainTumorClassificationModel/model",
    call_endpoint="serving_default"
)

def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_and_classify(image_bytes):
    image = preprocess_image_bytes(image_bytes)

    # Detection
    detect_pred = detection_model(image)
    has_tumor = bool(detect_pred["output_0"][0][0])

    if not has_tumor:
        return {
            "has_tumor": False,
            "tumor_type": "no tumor",
            "confidence": 100.0
        }

    # Classification
    class_pred = classification_model(image)["output_0"]
    class_pred = class_pred.numpy()
    idx = int(np.argmax(class_pred))

    return {
        "has_tumor": True,
        "tumor_type": CLASS_NAMES[idx],
        "confidence": float(np.max(class_pred) * 100)
    }