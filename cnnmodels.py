from keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import io

CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Load models ONCE


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

def classify_tumor(image_bytes):
    # 1. Preprocess the image
    image = preprocess_image_bytes(image_bytes)

    # 2. Run Classification directly
    # Removed the detection_model block entirely
    class_pred_tensor = classification_model(image)["output_0"]
    class_pred = class_pred_tensor.numpy()
    
    # 3. Get the index of the highest confidence score
    idx = int(np.argmax(class_pred))
    confidence = float(np.max(class_pred) * 100)
    tumor_type = CLASS_NAMES[idx]

    # 4. Determine has_tumor status based on the classification result
    # We assume "no tumor" is a string in your CLASS_NAMES list
    is_tumor_detected = tumor_type.lower() != "no tumor"

    return {
        "has_tumor": is_tumor_detected,
        "tumor_type": tumor_type,
        "confidence": confidence
    }
