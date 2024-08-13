from fastapi import FastAPI
from pydantic import BaseModel
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from fastapi.responses import JSONResponse
import pickle

app = FastAPI()

class SentenceRequest(BaseModel):
    sentence: str

# custom_objects = {"InputLayer": InputLayer}
# model = load_model('pos_ner_model.h5', custom_objects=custom_objects)
max_len = 31
tokenizer = Tokenizer()
# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('pos_ner_model.h5')
pos_tag_to_index = {'NNP': 0,'PUNCT': 1,'NNC': 2,'ADJ': 3,'DET': 4,'VF': 5,'CONJ': 6,'PRO': 7,'VNF': 8,'PP': 9,'QF': 10,'ADV': 11,'PART': 12,'OTH': 13,'INTJ': 14}
ner_tag_to_index = {'B-D&T': 0,'B-OTH': 1,'B-GPE': 2,'I-GPE': 3,'B-PER': 4,'I-PER': 5,'B-LOC': 6,'I-LOC': 7,'B-ORG': 8,'B-EVENT': 9,'I-EVENT': 10,'B-NUM': 11,'I-ORG': 12,'I-D&T': 13,'B-UNIT': 14,'B-MISC': 15,'B-T&T': 16,'I-NUM': 17,'I-T&T': 18,'I-MISC': 19,'I-UNIT': 20}
index_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_index.items()}
index_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_index.items()}

def preprocess_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def predict_pos_ner(sentence):
    preprocessed_sentence = preprocess_sentence(sentence)
    print(preprocessed_sentence)
    tokens = tokenizer.texts_to_sequences([preprocessed_sentence])
    tokens = pad_sequences(tokens, maxlen=max_len, padding='post')
    pos_pred, ner_pred = model.predict(tokens)
    pos_pred_indices = pos_pred.argmax(axis=-1)[0]
    ner_pred_indices = ner_pred.argmax(axis=-1)[0]
    pos_tags = [index_to_pos_tag.get(idx, 'O') for idx in pos_pred_indices]
    ner_tags = [index_to_ner_tag.get(idx, 'O') for idx in ner_pred_indices]
    words = tokenizer.sequences_to_texts(tokens)[0].split()
    result = [(word, pos_tag, ner_tag) for word, pos_tag, ner_tag in zip(words, pos_tags, ner_tags)]
    return result

@app.post("/predict/")
async def predict(sentence: str):
    result = predict_pos_ner(sentence)
    return {
        "sentence": sentence,
        "predictions": [
            {"word": word, "pos_tag": pos_tag, "ner_tag": ner_tag} for word, pos_tag, ner_tag in result
        ]
    }
  

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
