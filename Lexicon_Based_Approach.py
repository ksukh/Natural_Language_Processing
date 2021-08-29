# -*- coding: utf-8 -*-
"""
Submission to Question [1], Part [1]
"""

#Imports
import pandas as pd
import numpy as np
import random

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.metrics import accuracy_score, f1_score

#Import training set

df_train = pd.read_csv("sentiment_train.csv")

print(df_train.info())
print(df_train.head())

#Import test set
df_test = pd.read_csv("sentiment_test.csv")

print(df_test.info())
print(df_test.head())


print(df_train.Polarity.value_counts())
#The train data set has 2202 reviews sentences with imbalanced target label of positive and negative sentiment represented by 1 and 0 respectively.

#Train model using VADER
random.seed(30)
analyzer = SentimentIntensityAnalyzer()

def my_sentiment_analyzer(documents):
    """This function takes a list of documents and returns a list of corresponding sentiment predictions."""
 # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    preds = np.zeros(len(documents))

    for i, doc in enumerate(documents):
        sentiment_dict = sid_obj.polarity_scores(doc)

        if not sentiment_dict['neg'] >  0.3:
            if sentiment_dict['pos']-sentiment_dict['neg'] > 0:
                 preds[i] = 1
        if not sentiment_dict['pos'] >  0.3:
            if sentiment_dict['pos']-sentiment_dict['neg'] <= 0:
                 preds[i] = 0
    return preds

#Test model performance using Test set
pred_test = my_sentiment_analyzer(df_test['Sentence'])

print("Accuracy (test set) = {:.5f}".format(accuracy_score(df_test['Polarity'],pred_test)))
print("F1 score (test set) = {:.4f}".format(f1_score(df_test['Polarity'],pred_test)))

#Identify and create incorrect predictions dataframe
my_submission = pd.DataFrame({'Sentence': df_test.Sentence,'Polarity': df_test.Polarity, 'Predicted': pred_test.astype(int)})
didntmakeit = my_submission[my_submission['Predicted'] != my_submission['Polarity']]
didntmakeit.reset_index(drop=True, inplace=True)
print("Number of incorrect predictions: " + str(len(didntmakeit)))
#print(didntmakeit.head(5))

