
import os
import re
import time
import string
import itertools
import math
import nltk
import pandas as pd

from explore import *

def compute_kneserney_p_unigram(unigram_index, fdist_bigrams_df):
    p = len(fdist_bigrams_df[fdist_bigrams_df.ilast == unigram_index]) / len(fdist_bigrams_df)
    return(p)

def compute_kneserney_d(fdist_df):
    n1 = len(fdist_df[fdist_df.freq == 1])
    n2 = len(fdist_df[fdist_df.freq == 2])
    d = n1 / (n1 + 2 * n2)
    return(d)

def compute_kneserney_p_raw(fdist_df, model_d=None):

    if model_d is None:
        final_d = compute_knerseney_d(fdist_df)
    else:
        final_d = model_d

    if final_d < 0 or final_d > 1:
        raise ValueError("D should be between zero and one")

    p_raw = (fdist_df.freq - final_d) / fdist_df.freq.sum()
    return(p_raw)

def lower_order_ngram(root, last, sep=' '):
    lower_order_root = sep.join(root.split(sep)[1:])
    result = sep.join((lower_order_root, last))
    return(result)

def complete_sentence_simple_aux(ts, model, max_size):
    search_order = min(len(ts) + 1, model['order'])
    search_data_frame = model['ngrams'][search_order]
    if search_order == 1:
        return(['lastword'][0:max_size])
    search_root_dict = model['indices'][search_order - 1]
    root = ' '.join(ts)
    if root in search_root_dict:
        iroot = search_root_dict[root]
        if iroot in search_data_frame['iroot']:
            results = search_data_frame[search_data_frame.iroot == iroot]
            return(results[0:max_size])
        else:
            return(complete_sentence_simple_aux(ts[1:], model, max_size))
    

def complete_sentence_simple(root, model, max_size=5):
    print(root)
    ts = clean_and_tokenize(root)
    print(ts)
    ts = vocabulary_class_filter_list([ts], model['vocabulary'])[0]
    print(ts)
    if len(ts) < model['order'] - 1:
        ts = [tokens['sentence_start']] + ts
    if len(ts) > model['order'] - 1:
        ts = ts[len(ts) - (model['order'] - 1):len(ts)]
    print(ts)
    results = complete_sentence_simple_aux(ts, model, max_size)
    words = [model['word_index'][w] for w in results['ilast']]
    return([(w, p) for w, p in zip(words, results.p)])

def get_ngram_probability(ng, model):
    print(ng)
    p = 0.1;
    return(p)

def sentence_probability(sentence, model):
    tsentence = clean_and_tokenize(sentence)
    tsentence = vocabulary_class_filter_list([tsentence], model['vocabulary'])[0]
    p_sentence = tsentence_probability(tsentence, model)
    return(p_sentence)

def tsentence_probability(tsentence, model):
    ts = list(itertools.repeat(tokens['sentence_start'], model['order'] - 1))
    ts = ts + tsentence
    ts = ts + [tokens['sentence_end']]
    ng = ngrams(ts, model['order'])
    p_ng = [get_ngram_probability(n, model) for n in ng]
    p_sentence = functools.reduce(lambda x, y: x * y, p_ng)
    return(p_sentence)


def sentence_list_probability(sl, model):
    psl = [tsentence_probability(s, model) for s in sl]
    p = functools.reduce(lambda x, y: x * y, psl)
    return(p)

def sentence_list_probability_mp(sl, model):
    chunks = split_seq(sl, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(sentence_list_probability, chunks)
    p = functools.reduce(lambda x, y: x * y, map_results)
    return(p)

def get_cross_entropy(sl, model):
    p = sentence_list_probability(sl, model)
    number_of_words = sum([len(s) for s in sl]) + len(sl)
    h = -math.log2(p) / number_of_words
    return(h)

if __name__ == "__main__":
    start_time_script = time.time()

    print("Loading vocabulary and n-gram frequency data")
    start_time = time.time()
    print("   Vocabulary...")
    vocabulary_df = pd.read_csv(os.path.join(data_dir, "vocabulary.csv"), encoding='windows-1252')
    vocabulary_df.word = vocabulary_df.word.astype(str)
    model_vocabulary = set(vocabulary_df.word)
    print("   Unigrams...")
    fdist_unigrams_df = pd.read_csv(os.path.join(data_dir, "fdist_ngrams_1.csv"), encoding='windows-1252')
    fdist_unigrams_df.freq = fdist_unigrams_df.freq.astype(int)
    fdist_unigrams_df.lastword = fdist_unigrams_df.lastword.astype(str)
    print("   Bigrams...")
    fdist_bigrams_df = pd.read_csv(os.path.join(data_dir, "fdist_ngrams_2.csv"), encoding='windows-1252')
    fdist_bigrams_df.freq = fdist_bigrams_df.freq.astype(int)
    fdist_bigrams_df.lastword = fdist_bigrams_df.lastword.astype(str)
    fdist_bigrams_df.root = fdist_bigrams_df.root.astype(str)
    print("   Trigrams...")
    fdist_trigrams_df = pd.read_csv(os.path.join(data_dir, "fdist_ngrams_3.csv"), encoding='windows-1252')
    fdist_trigrams_df.freq = fdist_trigrams_df.freq.astype(int)
    fdist_trigrams_df.lastword = fdist_trigrams_df.lastword.astype(str)
    fdist_trigrams_df.root = fdist_trigrams_df.root.astype(str)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing model D")
    start_time = time.time()
    model_d = {}
    model_d[3] = compute_kneserney_d(fdist_trigrams_df)
    model_d[2] = compute_kneserney_d(fdist_bigrams_df)
    model_d[1] = compute_kneserney_d(fdist_unigrams_df)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Creating string to index dictionaries")
    start_time = time.time()
    model_dict_unigrams = dict([(u, i) for i, u in enumerate(fdist_unigrams_df.lastword)])
    z = zip(range(len(fdist_bigrams_df.index)), fdist_bigrams_df.root, fdist_bigrams_df.lastword)
    model_dict_bigrams = dict((' '.join((b_root, b_last)), i) for i, b_root, b_last in z)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Creating integer indices")
    start_time = time.time()
    fdist_unigrams_df['ilast'] = [model_dict_unigrams[w] for w in fdist_unigrams_df.lastword]

    fdist_bigrams_df['iroot'] = np.array([model_dict_unigrams[w] for w in fdist_bigrams_df.root], dtype=np.uint32)
    fdist_bigrams_df['ilast'] = np.array([model_dict_unigrams[w] for w in fdist_bigrams_df.lastword], dtype=np.uint32)

    fdist_trigrams_df['iroot'] = np.array([model_dict_bigrams[b] for b in fdist_trigrams_df.root], dtype=np.uint32)
    fdist_trigrams_df['ilast'] = np.array([model_dict_unigrams[w] for w in fdist_trigrams_df.lastword], dtype=np.uint32)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing unigram probabilities")
    start_time = time.time()
    g = fdist_bigrams_df[['ilast', 'lastword']].groupby(['ilast'])
    p_unigrams_df = pd.DataFrame({'ilast': g.ilast.first(), 'p': g.ilast.size() / len(fdist_bigrams_df)})
    p_unigrams_df = p_unigrams_df.reset_index(drop=True)
    p_unigrams = dict(zip(p_unigrams_df.ilast, p_unigrams_df.p))
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing unigram coefficients when bigram root")
    start_time = time.time()
    g = fdist_bigrams_df[['iroot', 'lastword']].groupby(['iroot'])
    rcoeff_unigrams_df = pd.DataFrame({'iroot': g.iroot.first(), 'rcoeff': model_d[2] * g.iroot.size() / fdist_bigrams_df.freq.sum()})
    rcoeff_unigrams_df = rcoeff_unigrams_df.reset_index(drop=True)
    rcoeff_unigrams = dict(zip(rcoeff_unigrams_df.iroot, rcoeff_unigrams_df.rcoeff))
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing smooth bigram probabilities")
    start_time = time.time()
    p_raw = compute_kneserney_p_raw(fdist_bigrams_df, model_d[2])
    rcoeff = np.array([rcoeff_unigrams[i] for i in fdist_bigrams_df.iroot])
    p_lower_model = np.array([p_unigrams[i] for i in fdist_bigrams_df.ilast])

    fdist_bigrams_df['p'] = p_raw + rcoeff * p_lower_model
    p_bigrams = dict((model_dict_bigrams[' '.join((root, last))], p) for root, last, p in zip(fdist_bigrams_df.root, fdist_bigrams_df.lastword, fdist_bigrams_df.p))
    print("Done. Took %f seconds" % (time.time() - start_time))


    print("Computing bigram coefficients when trigram root")
    start_time = time.time()
    g = fdist_trigrams_df[['iroot', 'lastword']].groupby(['iroot'])
    rcoeff_bigrams_df = pd.DataFrame({'iroot': g.iroot.first(), 'rcoeff': model_d[3] * g.iroot.size() / fdist_trigrams_df.freq.sum()})
    rcoeff_bigrams_df = rcoeff_bigrams_df.reset_index(drop=True)
    rcoeff_bigrams = dict(zip(rcoeff_bigrams_df.iroot, rcoeff_bigrams_df.rcoeff))
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing smooth trigram probabilities")
    start_time = time.time()
    p_raw = compute_kneserney_p_raw(fdist_trigrams_df, model_d[3])
    rcoeff = np.array([rcoeff_bigrams[i] for i in fdist_trigrams_df.iroot])
    # TODOOOO!!!!!!
    lower_model_indices = [model_dict_bigrams[lower_order_ngram(root, last)] for root, last in zip(fdist_trigrams_df.root, fdist_trigrams_df.lastword)]
    p_lower_model = np.array([p_bigrams[i] for i in lower_model_indices])

    fdist_trigrams_df['p'] = p_raw + rcoeff * p_lower_model
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Creating model")
    start_time = time.time()
    model = {}
    model['order'] = 3
    model['vocabulary'] = model_vocabulary
    model['indices'] = {1: model_dict_unigrams,
                        2: model_dict_bigrams}
    model['word_index'] = dict((i, w) for w, i in model_dict_unigrams.items())
    model_ngrams_1 = p_unigrams_df.sort_values(by='p', ascending=False)
    model_ngrams_2 = fdist_bigrams_df[['iroot', 'ilast', 'p']].sort_values(by='p', ascending=False)
    model_ngrams_3 = fdist_trigrams_df[['iroot', 'ilast', 'p']].sort_values(by='p', ascending=False)

    model['ngrams'] = {1: model_ngrams_1,
                       2: model_ngrams_2,
                       3: model_ngrams_3}

    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))
