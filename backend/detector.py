from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time

model = load_model("backend/cnn_model.h5")  # Ensure path is correct

def preprocess_image(path):
    image = load_img(path, target_size=(224, 224))  # Match model training size
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = image.reshape((1, 224, 224, 3))  # Required CNN shape
    return image

def detect_deepfake(image_path):
    img_array = preprocess_image(image_path)

    start_time = time.time()
    prediction = model.predict(img_array)[0][0]
    end_time = time.time()

    confidence = round(prediction * 100, 2)
    label = "Fake" if prediction > 0.5 else "Real"
    inference_time = round(end_time - start_time, 2)

    return label, confidence, inference_time
