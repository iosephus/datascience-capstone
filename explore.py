
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
from nltk import tokenize
import nltk.data

data_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Data"
text_encoding = "utf-8"
lang = "english"
max_word_len = 24
char_categories = {'alphanumeric': '[a-zA-Z]', 'space': r'[\s]', 'newline': r'\n', 'digit': '[0-9]', 'punctuation': re.escape(string.punctuation).join(['[', ']'])}
reduction_factor = 1.0
analyze_unigrams = True
analyze_bigrams = True
analyze_trigrams = True
analyze_quadrigrams = False
training_set_size = 0.8
num_proc = 10
vocabulary = set([w.lower() for w in words.words()])
min_freq = 50
token_sentence_start = '<s>'
token_sentence_end = '</s>'
token_num = '<num>'
token_unknown = '<unk>'

sw = nltk.corpus.stopwords.words("english")

def count_stopwords(ngram, sw=sw):
    if type(ngram) is str:
        return(sum([ng in sw for ng in ngram.split(' ')]))
    else:
        return(sum([ng in sw for ng in ngram]))

def split_seq(s, n):
    chunk_len = max(1, int(math.ceil(len(s) / n)))
    return([s[i:i + chunk_len] for i in range(0, len(s), chunk_len)])

def split_corpus_text(corpus_text, n):
    chunks_text = zip(*[split_seq(corpus_text[cat], n) for cat in corpus_text.keys()])
    chunks = [dict(zip(corpus_text.keys(), ct)) for ct in chunks_text]
    return(chunks)


def clean_text(text):
    #print('  Converting to lowercase')
    result = text.lower()
    #print('  Replacing space characters, newlines and dashes by spaces')
    result = re.sub('[-\n\t\r\f\v]', ' ', result)
    #print('  Removing contraction quotes')
    result = re.sub('([a-z0-9])\'([a-z])', r'\1\2', result)
    #print('  Shortening repeated strings')
    result = re.sub(r'([a-z])\1{3,}', r'\1\1\1', result)
    #print('  Removing punctuation')
    result = re.sub(re.escape(string.punctuation).join(['[', ']']), "", result)
    #print('  Removing repeated spaces')
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
    if result:
        return([item for item in result.split(' ') if item])
    else:
        return([])

def clean_and_tokenize_mp(text):
    result = clean_text_mp(text)
    return(result.split(' '))

def clean_and_tokenize_sentences(sentences):
    result = [clean_and_tokenize_mp(s) for s in sentences]
    result = [l for l in result if l]
    return(result)

def clean_and_tokenize_sentences_mp(sentences):
    chunks = split_seq(sentences, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(clean_and_tokenize_sentences, chunks)
    del(chunks)
    result = functools.reduce(lambda l1, l2: l1 + l2, map_results)
    result = [l for l in result if l]
    return(result)

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
        return(token_num)
    else:
        return(token_unknown)

def vocabulary_class_filter_list(sl, vocabulary=vocabulary):
    return([[vocabulary_class_filter(w, vocabulary) for w in s] for s in sl])

def vocabulary_class_filter_list_mp(sl):
    chunks = split_seq(sl, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(vocabulary_class_filter_list, chunks)
    del(chunks)
    result = functools.reduce(lambda l1, l2: l1 + l2, map_results)
    return(result)


def create_freq_dataframe(fdist, extra=None):
    fngrams, fcounts = list(zip(*fdist.items()))
    fngrams = [' '.join(t) for t in fngrams]
    df = pd.DataFrame({'ngram': fngrams, 'freq': fcounts})
    df['stopwords'] = np.array(gen_stopword_count_list_mp(fngrams))
    if extra is not None:
        fngrams_extra, fcounts_extra = list(zip(*extra.items()))
        df_extra = pd.DataFrame({'ngram': fngrams_extra, 'freq': fcounts_extra})
        df = df.append(df_extra)
    df.sort_values(by=['freq'], ascending=False, inplace=True)
    return(df)

def tokenize_into_sentences(lines):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    s = [sent_detector.tokenize(l) for l in lines]
    result = functools.reduce(lambda l1, l2: l1 + l2, s)
    return(result)

def tokenize_into_sentences_mp(all_lines):
    chunks = split_seq(all_lines, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(tokenize_into_sentences, chunks)
    del(chunks)
    print("Reducing text cleaning results")
    result = functools.reduce(lambda l1, l2: l1 + l2, map_results)
    return(result)

def get_ngrams(sentence, n, pad=True):
    if pad:
        padded_sentence = [token_sentence_start] + sentence + [token_sentence_end]
    else:
        padded_sentence = sentence
    result = list(ngrams(padded_sentence, n))
    return(result)

if __name__ == "__main__":
    start_time_script = time.time()

    print("Reading corpus")
    start_time = time.time()
    corpus_reader = nltk.corpus.PlaintextCorpusReader(data_dir, 'training_set\.txt', encoding=text_encoding)
    corpus_text = corpus_reader.raw()[0:90000000]
    print("Done. Took %f seconds" % (time.time() - start_time))

    #print("Splitting lines")
    #start_time = time.time()
    #lines = corpus_text.splitlines()
    #print("Done. Took %f seconds" % (time.time() - start_time))

    #print("Tokenizing into sentences (parallel)")
    #start_time = time.time()
    #sentences = tokenize_into_sentences_mp(lines)
    #sentences = [corpus_text]
    #print("Done. Took %f seconds" % (time.time() - start_time))

    print("Tokenizing into words (parallel)")
    start_time = time.time()
    tokens_unigrams_raw = clean_and_tokenize_mp(corpus_text)
    del(corpus_text)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing raw unigram frequencies")
    start_time = time.time()
    #unigrams_raw = functools.reduce(lambda l1, l2: l1 + l2, [list(ngrams(s, 1)) for s in tokens_unigrams_raw])
    unigrams_raw = list(ngrams(tokens_unigrams_raw, 1))
    fdist_unigrams_raw = nltk.FreqDist(unigrams_raw)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Filtering out-of-vocabulary unigrams")
    start_time = time.time()
    frequent_unigrams = set((unigram[0] for unigram, freq in fdist_unigrams_raw.items() if unigram[0].isalpha() and freq >= min_freq))
    vocabulary_extended = vocabulary.union(frequent_unigrams)
    tokens_unigrams = [vocabulary_class_filter(w, vocabulary_extended) for w in tokens_unigrams_raw]
    del(tokens_unigrams_raw)
    del(fdist_unigrams_raw)
    del(unigrams_raw)
    del(vocabulary_extended)
    del(frequent_unigrams)
    print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_unigrams:
        print("Computing unigram frequencies")
        start_time = time.time()
        #unigrams = functools.reduce(lambda l1, l2: l1 + l2, [get_ngrams(s, 1, pad=False) for s in tokens_unigrams])
        unigrams = ngrams(tokens_unigrams , 1)
        fdist_unigrams = nltk.FreqDist(unigrams)
        del(unigrams)
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
        #bigrams = functools.reduce(lambda l1, l2: l1 + l2, [get_ngrams(s, 2, pad=False) for s in tokens_unigrams])
        bigrams = ngrams(tokens_unigrams , 2)
        fdist_bigrams = nltk.FreqDist(bigrams)
        del(bigrams)
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
        #trigrams = functools.reduce(lambda l1, l2: l1 + l2, [get_ngrams(s, 3, pad=False) for s in tokens_unigrams])
        trigrams = ngrams(tokens_unigrams , 3)
        fdist_trigrams = nltk.FreqDist(trigrams)
        del(trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building trigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_trigrams_data = create_freq_dataframe(fdist_trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_trigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_3.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_trigrams)
        del(fdist_trigrams_data)

    if analyze_quadrigrams:
        print("Computing quadrigram frequencies")
        start_time = time.time()
        fdist_quadrigrams = nltk.FreqDist(ngrams(tokens_unigrams, 4))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building quadrigram frequencies data frame (parallel)")
        start_time = time.time()
        quadrigrams = functools.reduce(lambda l1, l2: l1 + l2, [get_ngrams(s, 4, pad=False) for s in tokens_unigrams])
        fdist_trigrams = nltk.FreqDist(quadrigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_quadrigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_4.csv"), index = False)

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))

