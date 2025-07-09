from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import io
import os, gdown, tensorflow as tf

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
DRIVE_FILE_ID = os.getenv("DRIVE_FILE_ID")

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
     ]

 # 38 class labels

# Define image preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict disease
def predict_disease(file: UploadFile):
    try:
        image_bytes = file.file.read()
        img = preprocess_image(image_bytes)

        # Predict
        preds = MODEL.predict(img)
        confidence = float(np.max(preds))  # ✅ Convert to native float

        predicted_class = CLASS_NAMES[np.argmax(preds)]

        # Optional rejection threshold
        if confidence < 0.75:
            return JSONResponse(content={
                "label": "Uncertain — Not a valid plant leaf image",
                "confidence": round(confidence * 100, 2)
            })

        return JSONResponse(content={
    "label": predicted_class,
    "confidence": round(confidence * 100, 2)
})


    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return JSONResponse(content={"error": "Prediction failed"}, status_code=500)