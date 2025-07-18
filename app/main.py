from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.predict import predict_disease

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",  # Vite local dev
    "https://plant-disease-detection-aastik.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Plant Disease Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = predict_disease(file)
    return result
