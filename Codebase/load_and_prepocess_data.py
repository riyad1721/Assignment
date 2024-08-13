import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data = data.reset_index()
    data = data.rename(columns={'level_0': 'word', 'level_1': 'pos','শনিবার (২৭ আগস্ট) রাতে পটুয়াখালী সদর থানার ভারপ্রাপ্ত কর্মকর্তা (ওসি) মো. মনিরুজ্জামান এ তথ্য নিশ্চিত করেছেন।': 'ner'})
    data.dropna(inplace=True)
    data['word'] = data['word'].str.strip()
    data['word'] = data['word'].str.replace(r'[^\w\s]', '', regex=True)
    return data

def tokenize_and_pad(data, max_len=None):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['word'].values)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(data['word'].values)
    if not max_len:
        max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, word_index, max_len

def map_tags_to_integers(data):
    pos_tags = list(data['pos'].unique())
    ner_tags = list(data['ner'].unique())
    pos_tag_to_index = {tag: idx for idx, tag in enumerate(pos_tags)}
    ner_tag_to_index = {tag: idx for idx, tag in enumerate(ner_tags)}
    data['pos'] = data['pos'].map(pos_tag_to_index)
    data['ner'] = data['ner'].map(ner_tag_to_index)
    return data,pos_tags,ner_tags, pos_tag_to_index, ner_tag_to_index

def pad_labels(data, max_len):
    pos_sequences = data['pos'].values.reshape(-1, 1)
    ner_sequences = data['ner'].values.reshape(-1, 1)
    pos_padded = pad_sequences(pos_sequences, maxlen=max_len, padding='post')
    ner_padded = pad_sequences(ner_sequences, maxlen=max_len, padding='post')
    return pos_padded, ner_padded
