
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
corpus_dir = os.path.join(data_dir, 'corpora', 'en_US')
corpus_file_pattern = ".+\.training\.txt"
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
num_proc = 12
vocabulary = set([w.lower() for w in words.words()])
min_freq = 50
token_sentence_start = '<s>'
token_sentence_end = '</s>'
token_number = '<num>'
token_ordinal = '<ord>'
token_email = '<email>'
token_url = '<url>'
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

def remove_non_contraction_single_quotes(text):
    result = re.sub('(^\'+)|(\'+$)', "", text)
    result = re.sub('([^a-zA-Z0-9])\'+', r'\1', result)
    result = re.sub('\'+([^a-zA-Z])', r'\1', result)
    return(result)

def remove_non_word_hyphens(text):
    result = text
    result = re.sub('([0-9])-([0-9])', r'\1\2', result)
    result = re.sub('(^-+)|(-+$)', "", text)
    result = re.sub('([^a-zA-Z0-9])-+', r'\1', result)
    result = re.sub('-+([^a-zA-Z0-9])', r'\1', result)
    return(result)

def clean_text(text):
    result = text
    #print('  Removing non-contraction quotes')
    result = remove_non_contraction_single_quotes(result)
    result = remove_non_word_hyphens(result)
    #print('  Shortening repeated strings')
    result = re.sub(r'([a-zA-Z])\1{3,}', r'\1\1\1', result)
    #print('  Removing non-word, space, hyphens, characters')
    result = re.sub('[^\s\w\'-]', '', result)
    return(result)

def clean_text_mp(text):
    chunks = split_seq(text, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(clean_text, chunks)
    print("Reducing text cleaning results")
    result = " ".join(map_results)
    return(result)

def clean_and_tokenize(text):
    result = clean_text(text)
    if result:
        return([item.lower() for item in re.split('[\s]+', result) if item])
    else:
        return([])

def clean_and_tokenize_mp(text):
    result = clean_text_mp(text)
    return(result.split(' '))

def clean_and_tokenize_sentences(sentences):
    result = [clean_and_tokenize(s) for s in sentences]
    result = [l for l in result if l]
    return(result)

def clean_and_tokenize_sentences_mp(sentences):
    chunks = split_seq(sentences, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(clean_and_tokenize_sentences, chunks)
    result = list(itertools.chain(*map_results))
    return(result)

def gen_stopword_count_list(ngrams_l):
    return([count_stopwords(text) for text in ngrams_l])

def gen_stopword_count_list_mp(ngrams_l):
    chunks = split_seq(ngrams_l, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(gen_stopword_count_list, chunks)
    result = list(itertools.chain(*map_results))
    return(result)

# Not sure if works, test!!!!
def parallelize_function(fun):
    def parfun(work):
        chunks = split_seq(work, num_proc)
        num_proc_final = len(chunks)
        with mp.Pool(num_proc_final) as p:
            map_results = p.map(fun, chunks)
        result = list(itertools.chain(*map_results))
        return(result)
    return(parfun)

def is_word_quick(w):
    return(re.match('^(?:[\w]+[\'-])*[\w]+$', w) is not None) 

def vocabulary_class_filter(w, vocabulary=vocabulary):
    stemmer = stem.snowball.EnglishStemmer()
    if is_word_quick(w) and (w in vocabulary or stemmer.stem(w) in vocabulary):
        return(w)
    elif re.match('^[0-9]*1st$|^[0-9]*2nd$|^[0-9]*3rd$|^[0-9]*[4567890]th$|^[0-9]*1[123]th$', w) is not None:
        return(token_ordinal)
    elif re.match('^[+-]?[0-9]+$', w) is not None:
        return(token_number)
    elif w == '_URL_':
        return(token_url)
    elif w == '_EMAIL_':
        return(token_email)
    else:
        return(token_unknown)

def vocabulary_class_filter_list(sl, vocabulary=vocabulary):
    return([[vocabulary_class_filter(w, vocabulary) for w in s] for s in sl])

def vocabulary_class_filter_list_mp(sl):
    chunks = split_seq(sl, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(vocabulary_class_filter_list, chunks)
    result = list(itertools.chain(*map_results))
    return(result)


def create_freq_dataframe(fdist, sep=' '):
    fngrams, fcounts = list(zip(*fdist.items()))
    lwords = [t[-1] for t in fngrams]
    df = pd.DataFrame({'lastword': lwords, 'freq': fcounts})
    if len(fngrams[0]) > 1:
        df['root'] = [sep.join(t[:-1]) for t in fngrams]
    df.sort_values(by=['freq'], ascending=False, inplace=True)
    return(df)

def tokenize_into_sentences(lines):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    s = [sent_detector.tokenize(l) for l in lines]
    result = list(itertools.chain(*s))
    return(result)

def tokenize_into_sentences_mp(all_lines):
    chunks = split_seq(all_lines, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(tokenize_into_sentences, chunks)
    result = list(itertools.chain(*map_results))
    return(result)

def get_ngrams(sentence, n, pad=True):
    if pad:
        padded_sentence = itertools.chain([token_sentence_start], sentence, [token_sentence_end])
    else:
        padded_sentence = sentence
    result = list(ngrams(padded_sentence, n))
    return(result)

def clean_corpus_from_urls_etc(text):
    result = text
    result = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '_URL_', result)
    result = re.sub('ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '_URL_', result)
    result = re.sub('([\w\-\.]+@\w[\w\-]+\.+[\w\-]+)', '_EMAIL_', result)
    return(result)


def get_ngram_prev(ngram, sep=' '):
    return(sep.join(ngram.split(sep)[:-1]))

def add_p_column(fdist, fdist_prev):
    freq_prev = dict(zip(fdist_prev['ngram'], fdist_prev['freq']))
    fdist['p'] = np.array(fdist['freq']) / np.array([freq_prev[get_ngram_prev(ngram)] for ngram in fdist['ngram']])

def compute_probabilities(fdist):

    fdist[1]['p'] = fdist[1]['freq'] / sum(fdist[1]['freq'])

    for n in sorted(fdist.keys())[1:]:
        fdist[1]['p'] = fdist[1]['freq'] / sum(fdist[1]['freq'])

def get_next_word(fdist, phrase, max_size=5, do_not_predict=[token_sentence_start, token_sentence_end, token_number, token_ordinal, token_email, token_url, token_unknown]):
    phrase_tokens = list(itertools.chain([token_sentence_start], clean_and_tokenize(phrase)))
    n_values = sorted(fdist.keys())
    n_min = min(n_values)
    n_max = max(n_values)
    start = min(len(phrase_tokens) + 1, n_max - 1)
    search_values = sorted([n for n in n_values if n >= start], reverse=True)
    def get_next_word_aux(remaining_search_values):
        n = remaining_search_values[0]
        if n == 1:
              return(fdist[1]['ngram'][0:max_size])
        root = ''.join([' '.join(phrase_tokens[1 - n:]), ' '])
        fdist_data = fdist[n]
        matches = fdist_data[fdist_data['ngram'].str.startswith(root)]['ngram'][0:max_size]
        if len(matches) > 0:
            return matches
        return(get_next_word_aux(remaining_search_values[1:]))
    results = get_next_word_aux(search_values)
    return(results)

def create_optimized_ngram_model(fdist, sep=' '):
    result = pd.DataFrame(freq=df['freq'])
    if order == 1:
        result['lastword'] = [text_dicts[1][s] for s in df['ngram']]
    else:
        splits = [s.split(sep) for s in df['ngram']]
        result['lastword'] = [text_dicts[1][s[-1]] for s in splits]
        result['root'] = [sep.join(s[:-1]) for s in splits]
    return(result)

def create_model(fdist):
    text_dicts = dict([(k, dict(v['ngram'], range(len(v['ngram'])))) for k, v in fdist.items()])

if __name__ == "__main__":
    start_time_script = time.time()

    print("Reading corpus")
    start_time = time.time()
    corpus_reader = nltk.corpus.PlaintextCorpusReader(corpus_dir, corpus_file_pattern, encoding=text_encoding)
    corpus_text = corpus_reader.raw()
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Removing URLs, emails, etc.")
    start_time = time.time()
    corpus_text = clean_corpus_from_urls_etc(corpus_text)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Splitting lines (sentences)")
    start_time = time.time()
    lines = corpus_text.splitlines()
    print("Done. Took %f seconds" % (time.time() - start_time))
    del(corpus_text)

    print("Tokenizing into sentences (parallel)")
    start_time = time.time()
    sentences = tokenize_into_sentences_mp(lines)
    print("Done. Took %f seconds" % (time.time() - start_time))
    del(lines)

    print("Tokenizing into words (parallel)")
    start_time = time.time()
    tokens_unigrams_raw = clean_and_tokenize_sentences_mp(sentences)
    print("Done. Took %f seconds" % (time.time() - start_time))
    del(sentences)

    print("Computing raw unigram frequencies")
    start_time = time.time()
    unigrams_raw = list(itertools.chain(*[list(ngrams(s, 1)) for s in tokens_unigrams_raw]))
    fdist_unigrams_raw = nltk.FreqDist(unigrams_raw)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Filtering out-of-vocabulary unigrams")
    start_time = time.time()
    frequent_unigrams = set((unigram[0] for unigram, freq in fdist_unigrams_raw.items() if is_word_quick(unigram[0]) and freq >= min_freq))
    vocabulary_extended = vocabulary.union(frequent_unigrams)
    tokens_unigrams = vocabulary_class_filter_list(tokens_unigrams_raw, vocabulary_extended)
    print("Done. Took %f seconds" % (time.time() - start_time))
    del(tokens_unigrams_raw)
    del(unigrams_raw)
    
    if analyze_unigrams:
        print("Computing unigram frequencies")
        start_time = time.time()
        unigrams = list(itertools.chain(*[get_ngrams(s, 1, pad=True) for s in tokens_unigrams]))
        fdist_unigrams = nltk.FreqDist(unigrams)
        fdist_unigrams = dict([(unigram, freq) for unigram, freq in fdist_unigrams.items() if unigram[0] != token_sentence_end])
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building unigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_unigrams_data = create_freq_dataframe(fdist_unigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_unigrams)

        print("Saving unigram frequencies")
        start_time = time.time()
        fdist_unigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_1.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_bigrams:
        print("Computing bigram frequencies")
        start_time = time.time()
        bigrams = list(itertools.chain(*[get_ngrams(s, 2, pad=True) for s in tokens_unigrams]))
        fdist_bigrams = nltk.FreqDist(bigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building bigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_bigrams_data = create_freq_dataframe(fdist_bigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_bigrams)

        print("Saving bigram frequencies")
        start_time = time.time()
        fdist_bigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_2.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_trigrams:
        print("Computing trigram frequencies")
        start_time = time.time()
        trigrams = list(itertools.chain(*[get_ngrams(s, 3, pad=True) for s in tokens_unigrams]))
        fdist_trigrams = nltk.FreqDist(trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building trigram frequencies data frame (parallel)")
        start_time = time.time()
        fdist_trigrams_data = create_freq_dataframe(fdist_trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_trigrams)

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_trigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_3.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))

    if analyze_quadrigrams:
        print("Computing quadrigram frequencies")
        start_time = time.time()
        fdist_quadrigrams = nltk.FreqDist(ngrams(tokens_unigrams, 4))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building quadrigram frequencies data frame (parallel)")
        start_time = time.time()
        quadrigrams = list(itertools.chain(*[get_ngrams(s, 4, pad=True) for s in tokens_unigrams]))
        fdist_quadrigrams = nltk.FreqDist(quadrigrams)
        fdist_quadrigrams_data = create_freq_dataframe(fdist_quadrigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_quadrigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_4.csv"), index = False)

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))

