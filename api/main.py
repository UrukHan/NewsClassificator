# Import libraries
from api import model
from fastapi import FastAPI
from dto import dto

# Instantiating the model
model = model.BERT_MODEL()

# Determining how queries work
app = FastAPI()

@app.get("/health_check")
def health_check():
    return {"code": 200, "status": "OK"}

@app.post("/api/predict", response_model = dto.PredictionOut)
async def answer(user_request: dto.UserRequestIn):
    text = user_request.text
    prediction = model.predict([text])
    print(prediction)
    return {'output_text': prediction}
