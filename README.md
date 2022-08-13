# Spam detection

## General info

The project concerns spam detection in SMS messages to  determined whether the messages is spam or not. It includes data analysis, data preparation, text mining and create model by using different machine learning algorithms and **BERT model**. 

## The data
The dataset comes from SMS Spam Collection and can be find [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset). This SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It comprises one set of SMS messages in English of 5,574 messages, which is tagged acording being ham (legitimate) or spam.

## Motivation
The aim of the project is spam detection in SMS messages. One of the way to reduce the number of scams messages is spam filtering. For this purpose we may use different machine learning classifiers to sorted messages as a spam or not (such as Naive Bayes algorithm). In our analysis we used text classification with different machine learning algorithms to determined whether the messages is spam or not.  

## Project contains:
- Spam classification with ML algorithms - **Spam.ipynb**
- Spam classification with BERT model - **Spam_bert.ipynb**
- Python script to use spam model - **spam_model.py**
- Python script to use spam model with smote method - **spam_smote_model.py**

## Summary
We begin with data analysis and data pre-processing from our dataset. Following we used NLP methods to prepare and clean our text data (tokenization, remove stop words, stemming). In the first approach we used bag of words model to convert the text into numerical feature vectors. To get more accurate predictions we have applied six different classification algorithms like: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest, Stochastic Gradient Descent and Gradient Boosting. Finally we got the best accuracy of 97 % for Naive Bayes method. 

In the second we have used a pretrained BERT model to resolve our problem. In our second analysis we have used a Huggingface Transformers library as well. We have used a simple neural network with pretrained BERT model. We achieved an accuracy on the test set equal to 98 % and it is a very good result in comparison to previous models. 
From our experiments we can see that the both tested approaches give an overall high accuracy and similar results for our problem.

Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
**BERT** | Bert tokenizer | **0.98**
Naive Bayes| BOW | 0.97
Gradient Boosting| BOW | 0.96
Logistic Regression | BOW | 0.96
SVM | BOW  | 0.94
SGD |BOW  | 0.94
Random Forest | BOW | 0.92

## Technologies
#### The project is created with:

- Python 3.6
- libraries: pandas, numpy, scikit-learn, NLTK, imbalanced-learn, tensorflow, Hugging Face transformers.

#### Running the project:

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    spam_model.py
    spam_smote_model.py

