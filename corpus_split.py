
import os
import re
import time
import string
import itertools
import math
import nltk
import nltk.corpus
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools
import random
import codecs

from common import *

corpus_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Corpus\\Coursera-SwiftKey\\final\\en_US"
char_categories = {'alphanumeric': '[a-zA-Z]', 'space': r'[\s]', 'newline': r'\n', 'digit': '[0-9]', 'punctuation': re.escape(string.punctuation).join(['[', ']'])}
training_set_size = 0.8
random_seed = 34093610
training_set_filename = 'training_set.txt'
testing_set_filename = 'test_set.txt'
analyze_composition = False

def split_seq(s, n):
    chunk_len = max(1, int(math.ceil(len(s) / n)))
    return([s[i:i + chunk_len] for i in range(0, len(s), chunk_len)])

def get_text_character_composition(text, categories=char_categories):
    cat_items = categories.items()
    counts = [(cat_name, len(re.findall(cat_re, text))) for cat_name, cat_re in cat_items]
    return(dict(counts))

def get_text_character_composition_mp(text, categories):
    chunks = split_seq(text, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(get_text_character_composition, chunks)
    del(chunks)
    result = functools.reduce(lambda d1, d2: dict([(k, d1[k] + d2[k]) for k in d1]) , map_results)
    return(result)

def get_corpus_char_composition(corpus_text, categories=char_categories, normalize=False):
    fileids = corpus_text.keys()
    cat_items = categories.items()
    t3 = [(fid, cat_t[0], cat_t[1]) for (fid, cat_t) in itertools.product(fileids, cat_items)]
    df_cols = list(zip(*[(fid, cat_name, len(re.findall(cat_re, corpus_text[fid]))) for (fid, cat_name, cat_re) in t3]))
    df = pd.DataFrame({'file': df_cols[0], 'category': df_cols[1], 'count': df_cols[2]})
    df = df.pivot(index = 'file', columns = 'category', values = 'count')
    df.columns.name = None
    len_cols = list(zip(*[(fid, len(corpus_text[fid])) for fid in corpus_text]))
    df_len = pd.DataFrame(index=len_cols[0], data={'chars': len_cols[1]})
    df = df.join(df_len)
    if normalize:
        df[list(categories.keys())] = df[list(categories.keys())].div(np.array(df.chars), axis=0)
    return(df)

def split_corpus_text(corpus_text, n):
    chunks_text = zip(*[split_seq(corpus_text[cat], n) for cat in corpus_text.keys()])
    chunks = [dict(zip(corpus_text.keys(), ct)) for ct in chunks_text]
    return(chunks)

def get_corpus_char_composition_mp(corpus_text, word_counts, normalize=False):
    chunks = split_corpus_text(corpus_text, num_proc)
    num_proc_final = num_proc
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(get_corpus_char_composition, chunks)
    del(chunks)
    df = functools.reduce(lambda df1, df2: df1.add(df2) , map_results)
    df_wc = pd.DataFrame(index=word_counts.keys(), data={'words': list(word_counts.values())})
    df = df.join(df_wc)
    if normalize:
        df[list(char_categories.keys())] = df[list(char_categories.keys())].div(np.array(df.chars), axis=0)
    return(df)

if __name__ == "__main__":
    start_time_script = time.time()

    print("Reading corpus")
    start_time = time.time()
    corpus_reader = nltk.corpus.PlaintextCorpusReader(corpus_dir, '.+\.txt', encoding=text_encoding)
    corpus_text = dict([(fid, corpus_reader.raw(fid)) for fid in corpus_reader.fileids()])
    print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_composition:
        print("Tokenizing unigrams (parallel)")
        start_time = time.time()
        tokens_unigrams_raw = dict([(fid, clean_and_tokenize_mp(text)) for fid, text in corpus_text.items()])
        word_counts = dict([(fid, len(l)) for fid, l in tokens_unigrams_raw.items()])
        tokens_unigrams_raw = functools.reduce(lambda l1, l2: l1 + l2, tokens_unigrams_raw.values())
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Analyzing corpus character composition (parallel)")
        start_time = time.time()
        corpus_composition_data = get_corpus_char_composition_mp(corpus_text, word_counts, normalize=False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        print("Saving corpus character composition")
        start_time = time.time()
        corpus_composition_data.to_csv(os.path.join(data_dir, "corpus_composition.csv"), index = True)
        print("Done. Took %f seconds" % (time.time() - start_time))

    print("Splitting corpus into training/testing")
    start_time = time.time()
    lines = list(filter(lambda l: len(l) > 0, functools.reduce(lambda l1, l2: l1 + l2, [text.splitlines() for text in corpus_text.values()])))
    num_lines = len(lines)
)    print("Found %d total non-empty lines in corpus" % num_lines)
    num_lines_training = round(training_set_size * num_lines)
    print("Using %d (%f) randomly selected lines for the training set" % (num_lines_training, training_set_size))
    print("Generating training indices")
    training_indexes = set(random.sample(range(num_lines), num_lines_training))
    print("Generated %d indices" % len(training_indexes))
    print("Calculating testing indices")
    testing_indexes = list([i for i in range(num_lines) if i not in training_indexes])
    print("There are %d testing indices" % len(testing_indexes))
    print("Joining training set")
    training_set = "\n".join([lines[i] for i in training_indexes])
    print("Joining testing set")
    testing_set = "\n".join([lines[i] for i in testing_indexes])
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Saving training set")
    start_time = time.time()
    with codecs.open(os.path.join(data_dir, training_set_filename), 'w', encoding=text_encoding) as fp:
        fp.write(training_set)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Saving testing set")
    start_time = time.time()
    with codecs.open(os.path.join(data_dir, testing_set_filename), 'w', encoding=text_encoding) as fp:
        fp.write(testing_set)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))

