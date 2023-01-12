 # Load libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


stop_words = stopwords.words('english')
porter = PorterStemmer()

URL = '\data\spam.csv'


def clean_data(df):
    """Function to clean data"""
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
    df['Class'] = df['Class'].map({'ham': 0, 'spam': 1})
    return df


def text_preprocess(text):
    ''' The function to remove punctuation,
    stopwords and apply stemming'''
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in words.split() if word.lower()
             not in stop_words]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def read_data(path):
    ''' Function to read text data'''
    data = pd.read_csv(path, encoding='latin-1')
    dataset = clean_data(data)
    dataset['Text'] = data['Text'].apply(text_preprocess)
    X = dataset['Text']
    y = dataset['Class']
    return X, y


def prepare_data(X, y):
    ''' Function to split data on train and test set '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def create_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with score '''
    models = pd.DataFrame()
    classifiers = [
        LogisticRegression(),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=50),
        GradientBoostingClassifier(random_state=100, n_estimators=150,
                               min_samples_split=100, max_depth=6),
        LinearSVC(),
        SGDClassifier()]

    for classifier in classifiers:
        pipeline = imbpipeline(steps=[('vect', CountVectorizer(
                               min_df=5, ngram_range=(1, 2))),
                                      ('tfidf', TfidfTransformer()),
                                      ('smote', SMOTE()),
                                      ('classifier', classifier)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        param_dict = {
                     'Model': classifier.__class__.__name__,
                     'Score': score
                     }
        models = models.append(pd.DataFrame(param_dict, index=[0]))

    models.reset_index(drop=True, inplace=True)
    print(models.sort_values(by='Score', ascending=False))
   

if __name__ == '__main__':
    X, y = read_data(URL)
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    create_models(X_train, X_test, y_train, y_test)
