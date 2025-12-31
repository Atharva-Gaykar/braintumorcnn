from fastapi import FastAPI, UploadFile, File
from cnnmodels import detect_and_classify

app = FastAPI(title="Brain Tumor CNN Service")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = detect_and_classify(image_bytes)
    return result


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8500)