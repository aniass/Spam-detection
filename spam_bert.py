# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel


nltk.download('stopwords')
stop_words = stopwords.words('english')
porter = PorterStemmer()
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

URL = 'data\spam.csv'

# Constants
MAXLEN = 64
EPOCHS = 5


def clean_data(df):
    """Cleaning data and convert non numeric values"""
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
    df['Class'] = df['Class'].map({'ham': 0, 'spam': 1})
    return df


def text_preprocess(text):
    """Remove punctuation, stopwords and apply stemming"""
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in words.split() if word.lower()
             not in stop_words]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def read_data(path):
    """Read and preprocess data"""
    data = pd.read_csv(path, encoding='latin-1')
    dataset = clean_data(data)
    dataset['Text'] = data['Text'].apply(text_preprocess)
    return dataset


def prepare_data(data, test_size=0.2, random_state=42):
    """Spliting data into train and test set"""
    X = data['Text']
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def encode(text, maxlen=MAXLEN):
    """Encoding dataset with BERT tokenizer"""
    input_ids=[]
    attention_masks=[]

    for row in text:
        encoded = tokenizer.encode_plus(
            row,
            add_special_tokens=True,
            max_length=maxlen,
            pad_to_max_length=True,
            return_attention_mask=True,
        ) 
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


def build_model(input_shape=(64,), dense_units=32, dropout_rate=0.2):
    """Creating model using BERT"""
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    input_word_ids = tf.keras.Input(shape=input_shape,dtype='int32')
    attention_masks = tf.keras.Input(shape=input_shape,dtype='int32')

    sequence_output = bert_model([input_word_ids,attention_masks])
    output = sequence_output[1]
    output = tf.keras.layers.Dense(dense_units,activation='relu')(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs = [input_word_ids,attention_masks], outputs = output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train_input_ids, X_train_attention_masks, X_test_input_ids, 
                X_test_attention_masks, y_train, y_test):
    """Training the model"""
    history = model.fit(
        [X_train_input_ids, X_train_attention_masks],
        y_train,
        batch_size=32,
        epochs=EPOCHS,
        validation_data=([X_test_input_ids, X_test_attention_masks], y_test),
        class_weight= {0: 1, 1: 8})
    return history


def plot_graphs(history, string):
    """Visualization of training model"""
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  

def get_prediction(model, X_test_input_ids, X_test_attention_masks, y_test):
    """Getting predictions on a test set"""
    loss, accuracy = model.evaluate([X_test_input_ids, X_test_attention_masks], y_test)
    print('Test accuracy :', accuracy)
    return accuracy
    

if __name__ == '__main__':
    data = read_data(URL)
    X_train, X_test, y_train, y_test = prepare_data(data)
    X_train_input_ids, X_train_attention_masks = encode(X_train.values)
    X_test_input_ids, X_test_attention_masks = encode(X_test.values)
    model = build_model()
    history = train_model(model, X_train_input_ids, X_train_attention_masks,
                          X_test_input_ids, X_test_attention_masks, y_train, y_test)
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")
    get_prediction(model, X_test_input_ids, X_test_attention_masks, y_test)
