from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)


MODELSPATH = 'C:\Python Scripts\Projects_done\Spam\spam_best_model.pkl'

stop_words = stopwords.words('english')
porter = PorterStemmer()


def load_model():
    '''Loading pretrained model'''
    with open(MODELSPATH, 'rb') as file:
        model = load(file)
        return model
    

def preprocess_data(text):
    ''' Applying stopwords and stemming on raw data'''
    words = [word.lower() for word in text if word.lower() not in stop_words]
    words = [porter.stem(word) for word in words]
    return words


def get_prediction(input_text):
    ''' Generating predictions from raw data'''
    model = load_model()
    data = [input_text]
    text =  preprocess_data(data)
    result = model.predict(text)
    if result == 1:
        prediction = 'spam'
    else:
        prediction = 'not spam'
    print('---------------')
    print(f'Your message is {prediction}')


if __name__ == '__main__':
    text = input("Type a your message:\n")
    get_prediction(text)
    