# -*- coding: utf-8 -*-
"""
Submission to Question [2], Part [1]
"""


#Imports
import pandas as pd
import numpy as np
import string
import time

import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from nltk.sentiment.util import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


#Import training set
ml_train = pd.read_csv("sentiment_train.csv")

print(ml_train.info())
print(ml_train.head())

#Check target label count. 1 is positive sentiment and 0 is negative sentiment
print(ml_train.Polarity.value_counts())

#Import test set
ml_test = pd.read_csv("sentiment_test.csv")

print(ml_test.info())
print(ml_test.head())

random.seed(30)
#Text Pre-processing (same pre-processing steps are done for both train and test sets)
## Case normalization
ml_train['Sentence_lower'] = ml_train.Sentence.apply(lambda x:x.lower())
#print(ml_train.head())

ml_test['Sentence_lower'] = ml_test.Sentence.apply(lambda x:x.lower())
#print(ml_test.head())

## Remove punctuation
ml_train['Sentence_punc'] = ml_train['Sentence_lower'].str.replace("\W", ' ')
#print(ml_train.head())

ml_test['Sentence_punc'] = ml_test['Sentence_lower'].str.replace("\W", ' ')
#print(ml_test.head())

## Remove numbers
ml_train['Sentence_numbers'] = ml_train['Sentence_punc'].str.replace("\d+", '')
#print(ml_train.Sentence_numbers[28])

ml_test['Sentence_numbers'] = ml_test['Sentence_punc'].str.replace("\d+", '')

## Remove stop words
nltk.download('stopwords')
stop = stopwords.words('english')

ml_train['Sentence_stop'] = ml_train.Sentence_numbers.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#print(ml_train.head())

ml_test['Sentence_stop'] = ml_test.Sentence_numbers.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#print(ml_test.head())

## Tokenization
ml_train['Sentence_tokenized'] = ml_train.Sentence_stop.apply(lambda x: word_tokenize(x))
#print(ml_train.head())

ml_test['Sentence_tokenized'] = ml_test.Sentence_stop.apply(lambda x: word_tokenize(x))
#print(ml_test.head())

## Lemmatization
#Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices.
#It makes use of the vocabulary and does a morphological analysis to obtain the root word. Therefore, we usually prefer using lemmatization over stemming.
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

ml_train['Sentence_lemmatized'] = ml_train.Sentence_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
#print(ml_train.head())

ml_test['Sentence_lemmatized'] = ml_test.Sentence_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
#print(ml_test.head())

## Create a clean 'Sentence'
ml_train['Sentence_clean'] = ml_train.Sentence_lemmatized.apply(lambda x: ' '.join(x))
print(ml_train.head())

ml_test['Sentence_clean'] = ml_test.Sentence_lemmatized.apply(lambda x: ' '.join(x))
print(ml_test.head())

# Define Target Label
y_train = ml_train['Polarity']
X_train = ml_train.drop(['Polarity'], axis=1)

y_test = ml_test['Polarity']
X_test = ml_test.drop(['Polarity'], axis=1)

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

# Vectorization - Bag of Words with TDIDF
MAX_FEATURES=2000
NGRAM_RANGE=(1,3)
tdif_vectorizer = TfidfVectorizer(max_features = MAX_FEATURES, ngram_range=NGRAM_RANGE) #Vectorization

##Learn data vocabulary and create a document-term matrix
train_tdif = tdif_vectorizer.fit_transform(ml_train['Sentence_clean'])
test_tdif = tdif_vectorizer.transform(ml_test['Sentence_clean'])

#print(train_tdif.shape, test_tdif.shape)

train_bow = pd.DataFrame(train_tdif.toarray(), columns=tdif_vectorizer.get_feature_names(), index=ml_train.index) #BoW train set
df_train_bow = pd.concat([ml_train, train_bow], axis=1)

test_bow = pd.DataFrame(test_tdif.toarray(), columns=tdif_vectorizer.get_feature_names(), index=ml_test.index) #BoW test set
df_test_bow = pd.concat([ml_test, test_bow], axis=1)

## Drop all other columns and create X train and test sets
X_train_bow = df_train_bow.drop(
    columns=['Sentence', 'Polarity','Sentence_lower', 'Sentence_punc', 'Sentence_numbers', 'Sentence_stop',
             'Sentence_tokenized', 'Sentence_lemmatized', 'Sentence_clean'], axis=1)

#print(X_train_bow.head())

X_test_bow = df_test_bow.drop(
    columns=['Sentence', 'Polarity','Sentence_lower', 'Sentence_punc', 'Sentence_numbers', 'Sentence_stop',
             'Sentence_tokenized', 'Sentence_lemmatized','Sentence_clean'], axis=1)

#print(X_test_bow.head())

# Model Development
## Logistic Regression with GridSearchCV for hyperparameter tuning
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2')

max_iter=[100,110,120,130,140]
C = [1.0,1.5,2.0,2.5,5,5.5,10]
grid_values = dict(max_iter=max_iter,C=C)

grid_lr = GridSearchCV(LR, param_grid = grid_values, cv=5, n_jobs=-1)

start_time = time.time()
grid_lr.fit(X_train_bow, y_train)

print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best estimators: " + str(grid_lr.best_estimator_))

## Test model
y_pred_lr = grid_lr.predict(X_test_bow)

print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("LR Accuracy = {:.4f}".format(accuracy_score(y_test, y_pred_lr)))
print("LR F1 Score = {:.4f}".format(f1_score(y_test, y_pred_lr)))

# Results
test_results = pd.DataFrame({'Sentence': ml_test.Sentence,'Polarity': ml_test.Polarity, 'Predicted': y_pred_lr.astype(int)})
#print(test_results.head())

# Identify incorrect predictions. Create a dataframe with incorrect predictions
didntmakeit = test_results[test_results['Predicted'] != test_results['Polarity']]
didntmakeit.reset_index(drop=True, inplace=True)
print("Number of incorrect predictions: " + str(len(didntmakeit)))
#print(didntmakeit.head(5))
