
import os
import re
import time
import string
import nltk
import nltk.corpus
from nltk.util import ngrams
import pandas as pd
import numpy as np


corpus_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Corpus\\Coursera-SwiftKey\\final\\en_US"
data_dir = "C:\\Users\\JoseM\\Projects\\Capstone\\Data"
encoding = "UTF-8"
lang = "english"
max_word_len = 24
char_categories = {'alphanumeric': '[a-zA-Z]', 'space': r'[\s]', 'digit': '[0-9]', 'punctuation': re.escape(string.punctuation)}
reduction_factor = 1.0

sw = nltk.corpus.stopwords.words("english")

def count_stopwords(ngram, sw=sw):
    if type(ngram) is str:
        return(sum([ng in sw for ng in ngram.split(' ')]))
    else:
        return(sum([ng in sw for ng in ngram]))


def get_text_character_composition(text, categories):
    counts = [(cat_name, len(re.findall(cat_re, text))) for cat_name, cat_re in categories.items()]
    return(dict(counts))

def clean_and_tokenize(text ):
    print('  Converting to lowercase')
    result = text.lower()
    print('  Replacing space characters, newlines and dashes by spaces')
    result = re.sub('[-\n\t\r\f\v]', ' ', result)
    print('  Removing contraction quotes')
    result = re.sub('([a-z0-9])\'([a-z])', r'\1\2', result)
    print('  Removing repeated spaces')
    result = re.sub(' {2,}', ' ', result)
    print('  Shortening repeated strings')
    result = re.sub(r'(.)\1{4,}', r'\1\1\1', result)
    print('  Removing punctuation')
    result = re.sub(re.escape(string.punctuation), "", result)
    print('  Splitting into words')
    return(result.split(' '))

def create_freq_dataframe(fdist):
    fngrams, fcounts = list(zip(*fdist.items()))
    fngrams = [' '.join(t) for t in fngrams]
    freq = np.array(fcounts, dtype=np.float64)
    freq = freq / freq.sum()
    df = pd.DataFrame({'ngram': fngrams, 'freq': freq})
    df['stopwords'] = np.array([count_stopwords(text) for text in fngrams])
    df.sort_values(by=['freq'], ascending=False, inplace=True)
    return(df)

if __name__ == "__main__":
    start_time_script = time.time()

    print("Reading corpus")
    corpus_reader = nltk.corpus.PlaintextCorpusReader(corpus_dir, '.+\.txt')

    print("Tokenizing unigrams")
    start_time = time.time()
    corpus_text = corpus_reader.raw()

    if reduction_factor < 1.0:
        new_size = int(max(1e6, round(reduction_factor * len(corpus_text))))
        corpus_text = corpus_text[0:new_size]

    tokens_unigrams = [w for w in clean_and_tokenize(corpus_text) if w.isalnum() and len(w) <= max_word_len]
    print("Done. Took %f seconds" % (time.time() - start_time))

    if True:
        print("Computing unigram frequencies")
        start_time = time.time()
        fdist_unigrams = nltk.FreqDist(ngrams(tokens_unigrams, 1))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building unigram frequencies data frame")
        start_time = time.time()
        fdist_unigrams_data = create_freq_dataframe(fdist_unigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving unigram frequencies")
        start_time = time.time()
        fdist_unigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_1.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_unigrams)
        del(fdist_unigrams_data)

    if True:
        print("Computing bigram frequencies")
        start_time = time.time()
        fdist_bigrams = nltk.FreqDist(ngrams(tokens_unigrams, 2))
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Building bigram frequencies data frame")
        start_time = time.time()
        fdist_bigrams_data = create_freq_dataframe(fdist_bigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving bigram frequencies")
        start_time = time.time()
        fdist_bigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_2.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(fdist_bigrams)
        del(fdist_bigrams_data)

    if True:
        print("Computing trigram frequencies")
        start_time = time.time()
        fdist_trigrams = nltk.FreqDist(ngrams(tokens_unigrams, 3))
        print("Done. Took %f seconds" % (time.time() - start_time))
        del(tokens_unigrams)

        print("Building trigram frequencies data frame")
        start_time = time.time()
        fdist_trigrams_data = create_freq_dataframe(fdist_trigrams)
        print("Done. Took %f seconds" % (time.time() - start_time))

        print("Saving trigram frequencies")
        start_time = time.time()
        fdist_trigrams_data.to_csv(os.path.join(data_dir, "fdist_ngrams_3.csv"), index = False)
        print("Done. Took %f seconds" % (time.time() - start_time))

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))

