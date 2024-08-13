import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from load_and_prepocess_data import load_and_preprocess_data,tokenize_and_pad,map_tags_to_integers,pad_labels


def build_model(max_len, word_index, pos_tags, ner_tags):
    input = Input(shape=(max_len,)) # Define the input layer with shape equal to the maximum sequence length
    model = Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_len)(input)
    model = Dropout(0.3)(model) # # Dropout layer to prevent overfitting by randomly setting a fraction (0.3) 
    model = Bidirectional(LSTM(units=64, return_sequences=True))(model) # # Bidirectional LSTM layer units=64 means 64 LSTM units; return_sequences=True allows the LSTM to output the full sequence
    pos_output = TimeDistributed(Dense(len(pos_tags), activation="softmax"))(model) # POS tagging output
    ner_output = TimeDistributed(Dense(len(ner_tags), activation="softmax"))(model) # NER output
    model = Model(inputs=input, outputs=[pos_output, ner_output]) # Combine outputs into a single model
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy", "accuracy"])  # Two accuracies, one for each output)
    return model

def train_model(model, X_train, y_train_pos, y_train_ner, X_val, y_val_pos, y_val_ner, model_save_path):
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, [y_train_pos, y_train_ner], validation_data=(X_val, [y_val_pos, y_val_ner]), epochs=10, batch_size=32)
    model.save(model_save_path)
    return history

if __name__ == "__main__":   
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Using GPU:", physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Avoid OOM errors by allocating memory as needed
    else:
        print("No GPU found, using CPU.")
    data = load_and_preprocess_data('data.tsv')
    padded_sequences, word_index, max_len = tokenize_and_pad(data)
    data,pos_tags,ner_tags, pos_tag_to_index, ner_tag_to_index = map_tags_to_integers(data)
    pos_padded, ner_padded = pad_labels(data, max_len)
    #Train, Test, Validation 
    X_train, X_temp, y_train_pos, y_temp_pos, y_train_ner, y_temp_ner = train_test_split(padded_sequences, pos_padded, ner_padded, test_size=0.3, random_state=42)
    X_val, X_test, y_val_pos, y_test_pos, y_val_ner, y_test_ner = train_test_split(X_temp, y_temp_pos, y_temp_ner, test_size=0.5, random_state=42)
    #train the model and save it
    model = build_model(max_len, word_index, pos_tags, ner_tags)
    history = train_model(model, X_train, y_train_pos, y_train_ner, X_val, y_val_pos, y_val_ner, 'ner_pos_model.h5')
