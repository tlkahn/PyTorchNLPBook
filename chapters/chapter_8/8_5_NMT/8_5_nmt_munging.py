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

# %%
from argparse import Namespace
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

# %%
args = Namespace(
    source_data_path="data/nmt/eng-fra.txt",
    output_data_path="data/nmt/simplest_eng_fra.csv",
    perc_train=0.7,
    perc_val=0.15,
    perc_test=0.15,
    seed=1337
)

assert args.perc_test > 0 and (args.perc_test + args.perc_val + args.perc_train == 1.0)

# %%
with open(args.source_data_path) as fp:
    lines = fp.readlines()
    
lines = [line.replace("\n", "").lower().split("\t") for line in lines]

# %%
data = []
for english_sentence, french_sentence in lines:
    data.append({"english_tokens": word_tokenize(english_sentence, language="english"),
                 "french_tokens": word_tokenize(french_sentence, language="french")})

# %%
filter_phrases = (
    ("i", "am"), ("i", "'m"), 
    ("he", "is"), ("he", "'s"),
    ("she", "is"), ("she", "'s"),
    ("you", "are"), ("you", "'re"),
    ("we", "are"), ("we", "'re"),
    ("they", "are"), ("they", "'re")
)


# %%
data_subset = {phrase: [] for phrase in filter_phrases}
for datum in data:
    key = tuple(datum['english_tokens'][:2])
    if key in data_subset:
        data_subset[key].append(datum)

# %%
counts = {k: len(v) for k,v in data_subset.items()}
counts, sum(counts.values())

# %%
np.random.seed(args.seed)

dataset_stage3 = []
for phrase, datum_list in sorted(data_subset.items()):
    np.random.shuffle(datum_list)
    n_train = int(len(datum_list) * args.perc_train)
    n_val = int(len(datum_list) * args.perc_val)

    for datum in datum_list[:n_train]:
        datum['split'] = 'train'
        
    for datum in datum_list[n_train:n_train+n_val]:
        datum['split'] = 'val'
        
    for datum in datum_list[n_train+n_val:]:
        datum['split'] = 'test'
    
    dataset_stage3.extend(datum_list)    

# %%
# here we pop and assign into the dictionary, thus modifying in place
for datum in dataset_stage3:
    datum['source_language'] = " ".join(datum.pop('english_tokens'))
    datum['target_language'] = " ".join(datum.pop('french_tokens'))

# %%
nmt_df = pd.DataFrame(dataset_stage3)

# %%
nmt_df.head()

# %%
nmt_df.to_csv(args.output_data_path)
