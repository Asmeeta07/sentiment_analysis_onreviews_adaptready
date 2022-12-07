import pandas as pd
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

import re

import numpy as np


def add_words_dict(dict_, filename):
    with open(filename, "r", encoding="utf8") as f:
        for line in f.readlines():
            line = line.split(" ")

            try:
                dict_[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue


words = {}

add_words_dict(words, "glove.6B.50d.txt")

tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()


def convert_to_token_list(x):
    tokens = tokenizer.tokenize(x)
    lowercase_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercase_tokens]
    actual_tokens = [t for t in lemmatized_tokens if t in words]

    return actual_tokens


def convert_to_vectors(message, word_dict=words):
    converted_to_tokenlist = convert_to_token_list(message)

    converted_to_vectors = []

    for word in converted_to_tokenlist:
        if word not in word_dict:
            continue

        vector = word_dict[word]
        converted_to_vectors.append(vector)

    return np.array(converted_to_vectors, dtype=float)


# max_ = 810

from copy import deepcopy


def padding(X, desired_sequence_len=1000):
    x_copy = deepcopy(X)

    #     for i, x in enumerate(X):
    seq_len = X.shape[0]
    seq_len_diff = desired_sequence_len - seq_len

    pad = np.zeros(shape=(seq_len_diff, 50))

    x_copy = np.concatenate([x_copy, pad])
    x_copy = x_copy.reshape((1, x_copy.shape[0], x_copy.shape[1]))

    return np.array(x_copy).astype(float)


