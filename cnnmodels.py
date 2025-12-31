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