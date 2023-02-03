# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: nlpbook
#     language: python
#     name: nlpbook
# ---

import collections

# %%
import os
import re
import string
from argparse import Namespace

import nltk.data
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

# %%
args = Namespace(
    raw_dataset_txt="data/books/frankenstein.txt",
    window_size=5,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/books/frankenstein_with_splits.csv",
    seed=1337,
)

# %%
# Split the raw text book into sentences
nltk.download("punkt")
tokenizer = nltk.data.load(
    "../../../nltk_data/tokenizers/punkt/english.pickle"
)
with open(args.raw_dataset_txt) as fp:
    book = fp.read()
sentences = tokenizer.tokenize(book)

# %%
print(len(sentences), "sentences")
print("Sample:", sentences[100])


# %%
# Clean sentences
def preprocess_text(text):
    text = " ".join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


# %%
cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]

# %%
# Global vars
MASK_TOKEN = "<MASK>"

# %%
# Create windows
flatten = lambda outer_list: [
    item for inner_list in outer_list for item in inner_list
]
windows = flatten(
    [
        list(
            nltk.ngrams(
                [MASK_TOKEN] * args.window_size
                + sentence.split(" ")
                + [MASK_TOKEN] * args.window_size,
                args.window_size * 2 + 1,
            )
        )
        for sentence in tqdm_notebook(cleaned_sentences)
    ]
)

# Create cbow data
data = []
for window in tqdm_notebook(windows):
    target_token = window[args.window_size]
    context = []
    for i, token in enumerate(window):
        if token == MASK_TOKEN or i == args.window_size:
            continue
        else:
            context.append(token)
    data.append([" ".join(token for token in context), target_token])


# Convert to dataframe
cbow_data = pd.DataFrame(data, columns=["context", "target"])

# %%
# Create split data
n = len(cbow_data)


def get_split(row_num):
    if row_num <= n * args.train_proportion:
        return "train"
    elif (row_num > n * args.train_proportion) and (
        row_num <= n * args.train_proportion + n * args.val_proportion
    ):
        return "val"
    else:
        return "test"


cbow_data["split"] = cbow_data.apply(lambda row: get_split(row.name), axis=1)

# %%
cbow_data.head()

# %%
# Write split data to file
cbow_data.to_csv(args.output_munged_csv, index=False)
