# -*- coding: utf-8 -*-
"""
Submission to Question [3], Part [1]
"""

#Imports
import numpy as np
import math
import re
import pandas as pd
import matplotlib.pyplot as plt

import random

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from sklearn.metrics import classification_report, accuracy_score, f1_score

#Import training set
random.seed(30)
dl_train = pd.read_csv("sentiment_train.csv")

print(dl_train.info())
print(dl_train.head())

#Import test set
dl_test = pd.read_csv("sentiment_test.csv")

print(dl_test.info())
print(dl_test.head())


print(dl_train.Polarity.value_counts())
#The train data set has 2202 reviews sentences with imbalanced target label of positive and negative sentiment represented by 1 and 0 respectively.

random.seed(30)
#Text Pre-processing (same pre-processing steps are done for both train and test sets)
## Case normalization
dl_train['Sentence_lower'] = dl_train.Sentence.apply(lambda x:x.lower())
#print(dl_train.head())

dl_test['Sentence_lower'] = dl_test.Sentence.apply(lambda x:x.lower())
#print(dl_test.head())

## Remove punctuation
dl_train['Sentence_punc'] = dl_train['Sentence_lower'].str.replace("\W", ' ')
#print(dl_train.head())

dl_test['Sentence_punc'] = dl_test['Sentence_lower'].str.replace("\W", ' ')
#print(dl_test.head())

## Remove numbers
dl_train['Sentence_numbers'] = dl_train['Sentence_punc'].str.replace("\d+", '')
#print(dl_train.Sentence_numbers[28])

dl_test['Sentence_numbers'] = dl_test['Sentence_punc'].str.replace("\d+", '')
#print(dl_test.head())

## Create a clean Sentence
dl_train['Sentence_clean'] = dl_train.Sentence_numbers.apply(lambda x: ''.join(x))
dl_train = dl_train.drop(
    columns=['Sentence', 'Sentence_lower', 'Sentence_punc', 'Sentence_numbers'], axis=1)
#print(dl_train.head())

dl_test['Sentence_clean'] = dl_test.Sentence_numbers.apply(lambda x: ''.join(x))
dl_test.drop(
    columns=['Sentence', 'Sentence_lower', 'Sentence_punc', 'Sentence_numbers'], axis=1, inplace=True)
#print(dl_test.head())

# Define Target Label
train_clean = dl_train.Sentence_clean
train_label = dl_train.Polarity

test_clean = dl_test.Sentence_clean
test_label = dl_test.Polarity

## Tokenization
train_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(train_clean, target_vocab_size=2**16)
train_inputs = [train_tokenizer.encode(sentence) for sentence in train_clean]

test_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(test_clean, target_vocab_size=2**16)
test_inputs = [test_tokenizer.encode(sentence) for sentence in test_clean]

## Padding
train_MAX_LEN = max([len(sentence) for sentence in train_inputs])
train_inputs = tf.keras.preprocessing.sequence.pad_sequences(train_inputs, value=0, padding="post", maxlen=train_MAX_LEN)
#print(train_inputs)

test_MAX_LEN = max([len(sentence) for sentence in test_inputs])
test_inputs = tf.keras.preprocessing.sequence.pad_sequences(test_inputs, value=0, padding="post", maxlen=test_MAX_LEN)
#print(test_inputs)

# Model Building
class DCNN(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)

        self.embedding = layers.Embedding(vocab_size, emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, activation="relu")
        self.pool_1 = layers.GlobalMaxPool1D()

        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, activation="relu")
        self.pool_2 = layers.GlobalMaxPool1D()

        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, activation="relu")
        self.pool_3 = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")

        else:
            self.last_dense = layers.Dense(units=nb_classes, activation="softmax")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool_1(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool_1(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool_1(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output

# Model Configuration
VOCAB_SIZE = train_tokenizer.vocab_size

EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = len(set(train_label))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

# Model Training
Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="sparse_categorical_accuracy")

Dcnn.fit(train_inputs, train_label, batch_size=BATCH_SIZE, epochs=NB_EPOCHS)

# Model Evaluation
predicted_classes = Dcnn.predict(test_inputs)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

# Results
target_names = ["Class {}".format(i) for i in range(NB_CLASSES)]

print(classification_report(test_label, predicted_classes, target_names=target_names))
print("CNN Accuracy = {:.4f}".format(accuracy_score(test_label,predicted_classes)))
print("CNN F1 Score = {:.4f}".format(f1_score(test_label,predicted_classes)))

# Identify incorrect predictions
incorrect = np.where(predicted_classes!=test_label)[0]
print ("Length of incorrect predictions: "+str(len(incorrect)))
#print(incorrect)
