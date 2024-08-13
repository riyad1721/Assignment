import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from load_and_prepocess_data import load_and_preprocess_data,tokenize_and_pad,map_tags_to_integers,pad_labels

def evaluate_model(X_test, y_test_pos, y_test_ner, model, pos_tag_to_index, ner_tag_to_index):
    loss, pos_accuracy, ner_accuracy = model.evaluate(X_test, [y_test_pos, y_test_ner])
    print(f"Test Loss: {loss}")
    print(f"Test POS Accuracy: {pos_accuracy}")
    print(f"Test NER Accuracy: {ner_accuracy}")
    pos_pred, ner_pred = model.predict(X_test)
    pos_pred_indices = pos_pred.argmax(axis=-1)
    ner_pred_indices = ner_pred.argmax(axis=-1)
    index_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_index.items()}
    index_to_ner_tag = {idx: tag for tag, idx in ner_tag_to_index.items()}

    pos_pred_tags = [[index_to_pos_tag.get(idx, 'O') for idx in sentence] for sentence in pos_pred_indices]
    ner_pred_tags = [[index_to_ner_tag.get(idx, 'O') for idx in sentence] for sentence in ner_pred_indices]
    pos_true_tags = [[index_to_pos_tag.get(idx, 'O') for idx in sentence] for sentence in y_test_pos]
    ner_true_tags = [[index_to_ner_tag.get(idx, 'O') for idx in sentence] for sentence in y_test_ner]
    pos_pred_flat = [tag for sublist in pos_pred_tags for tag in sublist]
    ner_pred_flat = [tag for sublist in ner_pred_tags for tag in sublist]
    pos_true_flat = [tag for sublist in pos_true_tags for tag in sublist]
    ner_true_flat = [tag for sublist in ner_true_tags for tag in sublist]
    print("POS Tagging Classification Report:")
    print(classification_report(pos_true_flat, pos_pred_flat))
    print("NER Tagging Classification Report:")
    print(classification_report(ner_true_flat, ner_pred_flat))

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Using GPU:", physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Avoid OOM errors by allocating memory as needed
    else:
        print("No GPU found, using CPU.")
    data = load_and_preprocess_data('data.tsv')
    padded_sequences, word_index,  max_len = tokenize_and_pad(data)
    data, pos_tags, ner_tags, pos_tag_to_index, ner_tag_to_index = map_tags_to_integers(data)
    pos_padded, ner_padded = pad_labels(data, max_len)
    model = tf.keras.models.load_model('ner_pos_model.h5')
    X_train, X_temp, y_train_pos, y_temp_pos, y_train_ner, y_temp_ner = train_test_split(padded_sequences, pos_padded, ner_padded, test_size=0.3, random_state=42)
    X_val, X_test, y_val_pos, y_test_pos, y_val_ner, y_test_ner = train_test_split(X_temp, y_temp_pos, y_temp_ner, test_size=0.5, random_state=42)
    evaluate_model(X_test, y_test_pos, y_test_ner, model, pos_tag_to_index, ner_tag_to_index)
