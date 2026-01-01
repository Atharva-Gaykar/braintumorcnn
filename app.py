from fastapi import FastAPI, UploadFile, File
from cnnmodels import classify_tumor

app = FastAPI(title="Brain Tumor CNN Service")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = classify_tumor(image_bytes)
    return result
