# Load libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

stop_words = stopwords.words('english')
porter = PorterStemmer()


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


# Load dataset
url = 'C:\\Python Scripts\\Datasets\\spam.csv'
df = pd.read_csv(url, encoding='latin-1')

clean_data(df)

# shape
print(df.shape)
print(df.head())

# Separate into input and output columns
X = df['Text']
y = df['Class']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Create models
models = pd.DataFrame()

classifiers = [
    LogisticRegression(),
    MultinomialNB(),
    RandomForestClassifier(n_estimators=50),
    GradientBoostingClassifier(random_state=100, n_estimators=150,
                               min_samples_split=100, max_depth=6),
    LinearSVC(),
    SGDClassifier()
    ]

for classifier in classifiers:
    pipe = Pipeline(steps=[('vect', CountVectorizer(
            tokenizer=text_preprocess, min_df=5, ngram_range=(1, 2))),
                          ('tfidf', TfidfTransformer()),
                          ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    param_dict = {
                  'Model': classifier.__class__.__name__,
                  'Score': score
                  }
    models = models.append(pd.DataFrame(param_dict, index=[0]))

models.reset_index(drop=True, inplace=True)
print(models.sort_values(by='Score', ascending=False))