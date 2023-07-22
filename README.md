# Spam detection

## General info

The project concerns spam detection in SMS messages to  determined whether the messages is spam or not. It includes data analysis, data preparation, text mining and create model by using different machine learning algorithms and **BERT model**. 

## The data
The dataset comes from SMS Spam Collection and can be find [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset). This SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It comprises one set of SMS messages in English of 5,574 messages, which is tagged acording being ham (legitimate) or spam.

## Motivation
The aim of the project was spam detection in SMS messages. The spam filtering is one of the way to reduce the number of scams messages. In the analyze was applied text classification with different machine learning algorithms (such as Naive Bayes, Logistic Regression, SVM, Random Forest) to determined whether the messages is spam or not. 

## Project contains:
- Spam classification with ML algorithms - **Spam.ipynb**
- Spam classification with BERT model - **Spam_bert.ipynb**
- Python script to use spam model - **spam_model.py**
- Python script to use spam model with smote method - **spam_smote_model.py**
- Python script to use spam model with BERT - **spam_bert.py**
- Python script to generate predictions from trained model - **predictions.py**
- models - models used in the project.

## Summary
The goal of the project was spam detection in SMS messages. I began with data analysis and data pre-processing from the dataset. Following I have used NLP methods to prepare and clean our text data (tokenization, remove stop words, stemming). In the first approach I have used bag of words model to convert the text into numerical feature vectors. To get more accurate predictions I have applied six different classification algorithms like: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest, Stochastic Gradient Descent and Gradient Boosting. Finally I  got the best accuracy of 97 % for Naive Bayes method. 

In the second I have used a pretrained BERT model from Huggingface Transformers library to resolve the problem. I have used a simple neural network with pretrained BERT model and I achieved an accuracy on the test set equal to 98 % and it is a very good result in comparison to previous models. From the experiments one can see that the both tested approaches give an overall high accuracy and similar results for the problem.

Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
**BERT** | Bert tokenizer | **0.98**
LinearSVC| BOW | 0.98
SGD |BOW  | 0.98
Random Forest | BOW | 0.98
Logistic Regression | BOW | 0.97
Gradient Boosting| BOW | 0.97
Naive Bayes | BOW | 0.95

## Technologies
#### The project is created with:

- Python 3.6
- libraries: pandas, numpy, scikit-learn, NLTK, imbalanced-learn, tensorflow, Hugging Face transformers.

#### Running the project:

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    spam_model.py
    spam_smote_model.py
    spam_bert.py
    predictions.py

