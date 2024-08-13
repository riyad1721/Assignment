import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from fastapi.responses import JSONResponse
import pickle
from load_and_prepocess_data import load_and_preprocess_data,map_tags_to_integers

max_len = 31
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
data = load_and_preprocess_data('data.tsv')
_,_,_, pos_tag_to_index, ner_tag_to_index = map_tags_to_integers(data)
index_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_index.items()}
index_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_index.items()}

def preprocess_sentence(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def predict_pos_ner(sentence, model):
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