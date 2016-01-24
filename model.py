
import os
import re
import time
import string
import itertools
import math
import nltk
import pandas as pd

from explore import *

model_cutoff = {}
model_cutoff[1] = 0
model_cutoff[2] = 0
model_cutoff[3] = 0

def compute_kneserney_p_unigram(unigram_index, fdist_bigrams_df):
    p = len(fdist_bigrams_df[fdist_bigrams_df.ilast == unigram_index]) / len(fdist_bigrams_df)
    return(p)

def compute_kneserney_d(fdist_df):
    n1 = len(fdist_df[fdist_df.freq == 1])
    n2 = len(fdist_df[fdist_df.freq == 2])
    d = n1 / (n1 + 2 * n2)
    return(d)

def compute_kneserney_d123(fdist_df):
    n1 = len(fdist_df[fdist_df.freq == 1])
    n2 = len(fdist_df[fdist_df.freq == 2])
    n3 = len(fdist_df[fdist_df.freq == 3])
    n4 = len(fdist_df[fdist_df.freq == 4])
    y = n1 / (n1 + 2 * n2)
    d1 = 1 - 2 * y * n2 / n1
    d2 = 2 - 3 * y * n3 / n2
    d3 = 3 - 4 * y * n4 / n3
    d123 = {1: d1, 2: d2, 3: d3, 'y': y}
    return(d123)

def get_d(count, model_d):
    return(model_d[min(count, 3)])

def compute_kneserney_p_discounted(fdist_df, freqsum, model_d):
    discounts = np.array([model_d[min(freq, 3)] for freq in fdist_df.freq])
    p_disc = (np.array(fdist_df.freq) - discounts) / freqsum
    p_disc = p_disc.clip(min=0)
    return(p_disc)

def compute_kneserney_rcoeff(fdist_df, freqsum, model_d, cutoff_corr=0):
    rcoeff = np.zeros(len(fdist_df))
    fdist_n1_df = fdist_df[fdist_df.freq == 1]
    if len(fdist_n1_df) > 0:
        g1 = fdist_n1_df[['iroot', 'lastword']].groupby(['iroot'])
        n1 = dict(zip(g1.iroot.first(), g1.iroot.size()))
        n1_arr = np.array([n1[i] if i in n1 else 0 for i in fdist_df.iroot])
        rcoeff = rcoeff + model_d[1] * n1_arr
    fdist_n2_df = fdist_df[fdist_df.freq == 2]
    if len(fdist_n2_df) > 0:
        g2 = fdist_n2_df[['iroot', 'lastword']].groupby(['iroot'])
        n2 = dict(zip(g2.iroot.first(), g2.iroot.size()))
        n2_arr = np.array([n2[i] if i in n2 else 0 for i in fdist_df.iroot])
        rcoeff = rcoeff + model_d[2] * n2_arr
    fdist_n3_df = fdist_df[fdist_df.freq >= 3]
    if len(fdist_n3_df) > 0:
        g3 = fdist_n3_df[['iroot', 'lastword']].groupby(['iroot'])
        n3 = dict(zip(g3.iroot.first(), g3.iroot.size()))
        n3_arr = np.array([n3[i] if i in n3 else 0 for i in fdist_df.iroot])
        rcoeff = rcoeff + (model_d[3] * n3_arr)
    rcoeff = rcoeff + cutoff_corr
    rcoeff = rcoeff / freqsum
    return(rcoeff)


def compute_knesserney_p_smooth(fdist_df, freqsum, model_d, fdist_lower_df, cutoff_corr=0):
    p_discounted = compute_kneserney_p_discounted(fdist_df, freqsum, model_d)
    r_coeff = compute_kneserney_rcoeff(fdist_df, freqsum, model_d, cutoff_corr)
    if 'root' in fdist_lower_df.columns:
        p_lower_dict = dict((' '.join((root, last)), p) for root, last, p in zip(fdist_lower_df.root, fdist_lower_df.lastword, fdist_lower_df.p))
        p_lower = np.array([p_lower_dict[lower_order_ngram(root, last)] for root, last in zip(fdist_df.root, fdist_df.lastword)])
    else:
        p_lower_dict = dict((last, p) for last, p in zip(fdist_lower_df.lastword, fdist_lower_df.p))
        p_lower = np.array([p_lower_dict[last] for last in fdist_df.lastword])
    p_smooth = p_discounted + r_coeff * p_lower
    return(p_smooth)

def lower_order_ngram(root, last, sep=' '):
    lower_order_root = sep.join(root.split(sep)[1:])
    result = sep.join((lower_order_root, last))
    return(result)

def complete_sentence_simple_aux(ts, model, max_size):
    search_order = min(len(ts) + 1, model['order'])
    search_data_frame = model['ngrams'][search_order]
    if search_order == 1:
        return(search_data_frame['lastword'][0:max_size])
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

def get_ngram_probability_aux(ng, model):
    search_order = len(ng)
    if search_order == 1:
        ng_index = model['indices'][1][ng[0]]
        return(model['ngram_p_dicts'][1][ng_index])
    # TODO!!!!!!!!!!!
    #ng_str = 
    ng_index = model['indices'][search_order][' '.join(ng)]


def get_ngram_probability(ng, model):
    if len(ng) > model['order']:
        raise ValueError("N-gram cannot be larger than model order")

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


def sentence_list_log_probability(sl, model):
    psl = [math.log2(tsentence_probability(s, model)) for s in sl]
    logp = functools.reduce(lambda x, y: x + y, psl)
    return(logp)

def sentence_list_probability_mp(sl, model):
    chunks = split_seq(sl, num_proc)
    num_proc_final = len(chunks)
    with mp.Pool(num_proc_final) as p:
        map_results = p.map(sentence_list_probability, chunks)
    p = functools.reduce(lambda x, y: x * y, map_results)
    return(p)

def get_cross_entropy(sl, model):
    logp = sentence_list_log_probability(sl, model)
    number_of_words = sum([len(s) for s in sl]) + len(sl)
    print("Number of words: %d" % number_of_words)
    h = -logp / number_of_words
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
    model_d[1] = compute_kneserney_d123(fdist_unigrams_df)
    model_d[2] = compute_kneserney_d123(fdist_bigrams_df)
    model_d[3] = compute_kneserney_d123(fdist_trigrams_df)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Applying cutoffs")
    start_time = time.time()
    freqsum_unigrams = fdist_unigrams_df.freq.sum()
    if model_cutoff[1] > 0:
        fdist_unigrams_df = fdist_unigrams_df[fdist_unigrams_df.freq > model_cutoff[1]]
        cutoff_corr_unigrams = freqsum_unigrams - fdist_unigrams_df.freq.sum()
    else:
        cutoff_corr_unigrams = 0

    freqsum_bigrams = fdist_bigrams_df.freq.sum()
    if model_cutoff[2] > 0:
        fdist_bigrams_df = fdist_bigrams_df[fdist_bigrams_df.freq > model_cutoff[2]]
        cutoff_corr_bigrams = freqsum_bigrams - fdist_bigrams_df.freq.sum()
    else:
        cutoff_corr_bigrams = 0

    freqsum_trigrams = fdist_trigrams_df.freq.sum()
    if model_cutoff[3] > 0:
        fdist_trigrams_df = fdist_trigrams_df[fdist_trigrams_df.freq > model_cutoff[3]]
        cutoff_corr_trigrams = freqsum_trigrams - fdist_trigrams_df.freq.sum()
    else:
        cutoff_corr_trigrams = 0
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Creating string to index dictionaries")
    start_time = time.time()
    model_dict_unigrams = dict([(u, i) for i, u in enumerate(fdist_unigrams_df.lastword)])
    z = zip(range(len(fdist_bigrams_df.index)), fdist_bigrams_df.root, fdist_bigrams_df.lastword)
    model_dict_bigrams = dict((' '.join((b_root, b_last)), i) for i, b_root, b_last in z)
    z = zip(range(len(fdist_trigrams_df.index)), fdist_trigrams_df.root, fdist_trigrams_df.lastword)
    model_dict_trigrams = dict((' '.join((b_root, b_last)), i) for i, b_root, b_last in z)
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
    fdist_unigrams_df['p'] = np.array([p_unigrams[i] if i in p_unigrams else 0 for i in fdist_unigrams_df.ilast])
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing smooth bigram probabilities")
    start_time = time.time()
    fdist_bigrams_df['p'] = compute_knesserney_p_smooth(fdist_bigrams_df, freqsum_bigrams, model_d[2], fdist_unigrams_df, cutoff_corr_bigrams)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Computing smooth trigram probabilities")
    start_time = time.time()
    fdist_trigrams_df['p'] = compute_knesserney_p_smooth(fdist_trigrams_df, freqsum_trigrams, model_d[3], fdist_bigrams_df, cutoff_corr_trigrams)
    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Creating model")
    start_time = time.time()
    model = {}
    model['order'] = 3
    model['vocabulary'] = model_vocabulary
    model['p0'] = 1.0 / len(model_vocabulary)
    model['indices'] = {1: model_dict_unigrams,
                        2: model_dict_bigrams}
    model['word_index'] = dict((i, w) for w, i in model_dict_unigrams.items())
    model_ngrams_1 = p_unigrams_df.sort_values(by='p', ascending=False)
    model_ngrams_1['p'] = np.array(model_ngrams_1['p'] * freqsum_unigrams, dtype=np.uint32)
    model_ngrams_2 = fdist_bigrams_df[fdist_bigrams_df.freq > 1][['iroot', 'ilast', 'p']].sort_values(by='p', ascending=False)
    model_ngrams_2['p'] = np.array(model_ngrams_2['p'] * freqsum_bigrams, dtype=np.uint32)
    model_ngrams_3 = fdist_trigrams_df[fdist_trigrams_df.freq > 1][['iroot', 'ilast', 'p']].sort_values(by='p', ascending=False)
    model_ngrams_3['p'] = np.array(model_ngrams_3['p'] * freqsum_trigrams, dtype=np.uint32)

    needed_bigram_indexes = set(model_ngrams_3.iroot)
    model_dict_bigrams2 = dict([(s, i) for s, i in model_dict_bigrams.items() if i in needed_bigram_indexes])

    model['ngrams'] = {1: model_ngrams_1,
                       2: model_ngrams_2,
                       3: model_ngrams_3}

    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Saving model")
    start_time = time.time()
    model_ngrams_1.to_csv(os.path.join(data_dir, "model_ngrams_1.csv"), index = False)
    model_ngrams_2.to_csv(os.path.join(data_dir, "model_ngrams_2.csv"), index = False)
    model_ngrams_3.to_csv(os.path.join(data_dir, "model_ngrams_3.csv"), index = False)

    model_dict_unigrams_df = pd.DataFrame({'str': list(model_dict_unigrams.keys()), 'index': list(model_dict_unigrams.values())})
    model_dict_unigrams_df.to_csv(os.path.join(data_dir, "model_dict_1.csv"), index = False)

    model_dict_bigrams_df = pd.DataFrame({'str': list(model_dict_bigrams.keys()), 'index': list(model_dict_bigrams.values())})
    model_dict_bigrams_df.to_csv(os.path.join(data_dir, "model_dict_2.csv"), index = False)

    print("Done. Took %f seconds" % (time.time() - start_time))

    print("Done with everything. Total time was %f seconds. Bye!" % (time.time() - start_time_script))
