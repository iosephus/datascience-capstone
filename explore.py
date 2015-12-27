
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


corpus_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Corpus\\Coursera-SwiftKey\\final\\en_US"
data_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Data"
encoding = "UTF-8"
lang = "english"
max_word_len = 24
char_categories = {'alphanumeric': '[a-zA-Z]', 'space': r'[\s]', 'digit': '[0-9]', 'punctuation': re.escape(string.punctuation).join(['[', ']'])}
reduction_factor = 1.0
analyze_unigrams = True
analyze_bigrams = True
analyze_trigrams = True
num_proc = 10
vocabulary = set([w.lower() for w in words.words()])
min_freq = 50

sw = nltk.corpus.stopwords.words("english")

def count_stopwords(ngram, sw=sw):
    if type(ngram) is str:
        return(sum([ng in sw for ng in ngram.split(' ')]))
    else:
        return(sum([ng in sw for ng in ngram]))

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

def get_corpus_char_composition_mp(corpus_text, normalize=False):
    chunks = split_corpus_text(corpus_text, num_proc)
    num_proc_final = num_proc
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(get_corpus_char_composition, chunks)
    del(chunks)
    df = functools.reduce(lambda df1, df2: df1.add(df2) , map_results)
    if normalize:
        df[list(char_categories.keys())] = df[list(char_categories.keys())].div(np.array(df.chars), axis=0)
    return(df)

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


def create_freq_dataframe(fdist):
    fngrams, fcounts = list(zip(*fdist.items()))
    fngrams = [' '.join(t) for t in fngrams]
    df = pd.DataFrame({'ngram': fngrams, 'freq': fcounts})
    df['stopwords'] = np.array(gen_stopword_count_list_mp(fngrams))
    df.sort_values(by=['freq'], ascending=False, inplace=True)
    return(df)

def get_reduced_text(text, reduction_factor, min_len=1e3):
    if reduction_factor <= 0 or reduction_factor > 1.0:
        raise ValueError("Reduction factor should be greater than zero and max one")
    if reduction_factor == 1.0:
        return(text)
    new_size = int(max(min_len, round(reduction_factor * len(text))))
    return(text[0:new_size])

#def reduce_fdist(fdist, vocabulary=vocabulary):
#    result = set()

if __name__ == "__main__":
    start_time_script = time.time()

    print("Reading corpus")
    start_time = time.time()
    corpus_reader = nltk.corpus.PlaintextCorpusReader(corpus_dir, '.+\.txt')
    corpus_text = dict([(fid, get_reduced_text(corpus_reader.raw(fid), reduction_factor)) for fid in corpus_reader.fileids()])
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Analyzing corpus character composition (parallel)")
    start_time = time.time()
    corpus_composition_data = get_corpus_char_composition_mp(corpus_text, normalize=True)
    print("Done. Took %f seconds" % (time.time() - start_time))
    print("Saving corpus character composition")
    start_time = time.time()
    corpus_composition_data.to_csv(os.path.join(data_dir, "corpus_composition.csv"), index = True)
    print("Done. Took %f seconds" % (time.time() - start_time))
    del(corpus_composition_data)

    print("Tokenizing unigrams (parallel)")
    start_time = time.time()
    tokens_unigrams_raw = clean_and_tokenize_mp('\n'.join(corpus_text.values()))
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing raw unigram frequencies")
    start_time = time.time()
    fdist_unigrams_raw = nltk.FreqDist(ngrams(tokens_unigrams_raw, 1))
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Filtering out-of-vocabulary unigrams")
    start_time = time.time()
    frequent_unigrams = set((unigram[0] for unigram, freq in fdist_unigrams_raw.items() if freq >= min_freq))
    vocabulary_extended = vocabulary.union(frequent_unigrams)
    tokens_unigrams = vocabulary_class_filter_list(tokens_unigrams_raw, vocabulary_extended)
    print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_unigrams:
        print("Computing unigram frequencies")
        start_time = time.time()
        fdist_unigrams = nltk.FreqDist(ngrams(tokens_unigrams, 1))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building unigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_unigrams_data = create_freq_dataframe(fdist_unigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving unigram frequencies")
        start_time = time.time()
        fdist_unigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_1.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_unigrams)
        del(fdist_unigrams_data)

    if analyze_bigrams:
        print("Computing bigram frequencies")
        start_time = time.time()
        fdist_bigrams = nltk.FreqDist(ngrams(tokens_unigrams, 2))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building bigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_bigrams_data = create_freq_dataframe(fdist_bigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving bigram frequencies")
        start_time = time.time()
        fdist_bigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_2.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_bigrams)
        del(fdist_bigrams_data)

    if analyze_trigrams:
        print("Computing trigram frequencies")
        start_time = time.time()
        fdist_trigrams = nltk.FreqDist(ngrams(tokens_unigrams, 3))
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(tokens_unigrams)

        print("Building trigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_trigrams_data = create_freq_dataframe(fdist_trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_trigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_3.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))

