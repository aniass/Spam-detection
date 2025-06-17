"""
Clean data for spam detection dataset
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

URL_DATA = 'data\spam.csv'


def read_data(path: str) -> pd.DataFrame:
    """Function to read data"""
    try:
        df = pd.read_csv(path, header=0, index_col=0)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to clean data"""
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
    df['Class'] = df['Class'].map({'ham': 0, 'spam': 1})
    return df


def clean_text(words):
    """The function to clean text"""
    words = re.sub("[^a-zA-Z]"," ", words)
    text = words.lower().split()                   
    return " ".join(text)


def remove_stopwords(text):
    """The function to removing stopwords"""
    stop_words = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(text)


def get_stemmer(stem_text):
    """The function to apply stemming"""
    porter = PorterStemmer()
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


# poprawic
def preprocess_data(data: str) -> str:
    """Function to preprocess data"""
    data['Text'] = data['Text'].apply(clean_text)
    data['Text'] = data['Text'].apply(remove_stopwords)
    data['Text'] = data['Text'].apply(get_stemmer)
    return data


if __name__ == '__main__':
    data = read_data(URL_DATA)
    dataset = clean_data(data)
    dataset = preprocess_data(dataset)
    if not dataset.empty:
        print(dataset.shape)
        print(dataset.head(5))
        dataset.to_csv(URL_DATA, encoding='utf-8')
