from fastapi import FastAPI
from tensorflow.keras.models import load_model
from inference import predict_pos_ner

model = load_model('ner_pos_model.h5')

app = FastAPI()


@app.post("/predict/")
async def predict(sentence: str):
    try:
        result = predict_pos_ner(sentence, model)
        return {"sentence": sentence,
                "predictions": [{"word": word, "pos_tag": pos_tag, "ner_tag": ner_tag} for word, pos_tag, ner_tag in result]}
    except:
        return {"status": "An exception occurs on prediction"}

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}