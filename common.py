
import os
import re
import time
import string
import itertools
import math
import nltk
import nltk.corpus
from nltk.util import ngrams
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools
from nltk.corpus import words
from nltk import stem

data_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Data"
text_encoding = "utf-8"
lang = "english"
num_proc = 10
vocabulary = set([w.lower() for w in words.words()])

def split_seq(s, n):
    chunk_len = max(1, int(math.ceil(len(s) / n)))
    return([s[i:i + chunk_len] for i in range(0, len(s), chunk_len)])

def split_corpus_text(corpus_text, n):
    chunks_text = zip(*[split_seq(corpus_text[cat], n) for cat in corpus_text.keys()])
    chunks = [dict(zip(corpus_text.keys(), ct)) for ct in chunks_text]
    return(chunks)

def clean_text(text):
    print('  Converting to lowercase')
    result = text.lower()
    print('  Replacing space characters, newlines and dashes by spaces')
    result = re.sub('[-\n\t\r\f\v]', ' ', result)
    print('  Removing contraction quotes')
    result = re.sub('([a-z0-9])\'([a-z])', r'\1\2', result)
    print('  Shortening repeated strings')
    result = re.sub(r'([a-z])\1{3,}', r'\1\1\1', result)
    print('  Removing punctuation')
    result = re.sub(re.escape(string.punctuation).join(['[', ']']), "", result)
    print('  Removing repeated spaces')
    result = re.sub(' {2,}', ' ', result)
    return(result)

def concat_lists_lowmem(list_of_lists):
    result = list_of_lists[0]
    for l in list_of_lists[1:]:
        result.extend(l)
        del(l[:])
    return(result)

def clean_text_mp(text):
    chunks = split_seq(text, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(clean_text, chunks)
    del(chunks)
    print("Reducing text cleaning results")
    result = " ".join(map_results)
    return(result)

def clean_and_tokenize(text):
    result = clean_text(text)
    return(result.split(' '))

def clean_and_tokenize_mp(text):
    result = clean_text_mp(text)
    return(result.split(' '))

def gen_stopword_count_list(ngrams_l):
    return([count_stopwords(text) for text in ngrams_l])

def gen_stopword_count_list_mp(ngrams_l):
    chunks = split_seq(ngrams_l, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(gen_stopword_count_list, chunks)
    del(chunks)
    result = functools.reduce(lambda l1, l2: l1 + l2, map_results)
    return(result)

def vocabulary_class_filter(w, vocabulary=vocabulary):
    stemmer = stem.snowball.EnglishStemmer()
    if w.isalpha() and (w in vocabulary or stemmer.stem(w) in vocabulary):
        return(w)
    elif w.isdigit() or re.match('^[0-9]*1st$|^[0-9]*2nd$|^[0-9]*3rd$|^[0-9]*[4567890]th$|^[0-9]*1[123]th$', w) is not None:
        return('<num>')
    else:
        return('<unk>')

def vocabulary_class_filter_list(wl, vocabulary=vocabulary):
    return([vocabulary_class_filter(w, vocabulary) for w in wl])

def vocabulary_class_filter_list_mp(wl):
    chunks = split_seq(wl, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(vocabulary_class_filter_list, chunks)
    del(chunks)
    result = functools.reduce(lambda l1, l2: l1 + l2, map_results)
    return(result)

