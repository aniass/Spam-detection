# Spam detection

## General info

The project concerns spam detection in SMS messages to  determined whether the messages is spam or not. It includes data analysis, data preparation, text mining and create model by using virtue different machine learning and BERT model. 

The dataset comes from SMS Spam Collection and can be find [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Motivation
The aim of the project is spam detection in SMS messages. We used text classification method to determined whether the messages is spam or not. First we used NLP methods to prepare and clean our text data (tokenization, remove stop words, stemming), then to get more accurate predictions we have applied different machine learning classification algorithms like: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest, Stochastic Gradient Descent and Gradient Boosting. In the second approach we have used a pretrained BERT model to resolve our problem.

## Project contains:
- Spam classification with ML algorithms - Spam.ipynb
- Spam classification with BERT model - Spam_bert.ipynb

## Technologies
#### The project is created with:

- Python 3.6
- libraries: pandas, numpy, scikit-learn, NLTK, imbalanced-learn, tensorflow, Hugging Face transformers.

#### Running the project:

To run this project use Jupyter Notebook or Google Colab.
