import nltk
from nltk.data import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import PAD_POS, MAX_CONTEXT_LEN
import numpy as np

nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def build_pos_indices():
    """
    Build pos2index and index2pos dictionaries of the UPenn POS tags
    """
    all_pos = [PAD_POS] + list(load("help/tagsets/upenn_tagset.pickle").keys())
    pos_to_idx = {pos: i for (i, pos) in enumerate(all_pos)}
    idx_to_pos = {i: pos for (i, pos) in enumerate(all_pos)}
    return pos_to_idx, idx_to_pos

def count_pos_tags():
    return len([PAD_POS] + list(load("help/tagsets/upenn_tagset.pickle").keys()))

def build_pos_features(df, maxlen):
    """
    Return the list of the categorical representation of token POS tags padded
    """
    _, idx_to_pos = build_pos_indices()
    doc_tags = [nltk.pos_tag(s) for s in df["proc_doc_tokens"]]
    return doc_tags, len(idx_to_pos.keys())

def exact_lemma(quest_tokens, doc_tokens):
    """
    Returns array of exact match and lemma features
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    exact = [1 if ecw in quest_tokens else 0 for ecw in doc_tokens]
    lemma = [1 if lemmatizer.lemmatize(et, pos="v") in quest_tokens else 0 for et in doc_tokens]
    bi = np.array([exact, lemma])
    return np.transpose(bi)

def term_frequency(tokens):
    """
    Returns term frequency of every word
    """
    l = len(tokens)
    tf = np.array([tokens.count(ew)/l for ew in tokens])
    return np.reshape(tf, (l,-1))

def build_exact_lemma_features(df, maxlen):
    """
    Computes and pad the exact lemma features for the questions and context columns of the dataframe given as argument
    """
    lemma_array = [exact_lemma(quest_tokens, doc_tokens) for (quest_tokens, doc_tokens) in zip(df["proc_quest_tokens"], df["proc_doc_tokens"])]  
    return pad_sequences(lemma_array, maxlen = maxlen, padding='post', truncating='post', value=np.array([0, 0]))

def build_term_frequency_features(df, maxlen):
    """
    Computes the term frequency features for the context column of the dataframe given as argument
    """
    tf = [term_frequency(tokens) for tokens in df["proc_doc_tokens"]]
    return pad_sequences(tf, maxlen = maxlen, padding='post', dtype='float64', truncating='post', value=0.0)

def get_additional_features(df, maxlen):
    doc_tags = build_pos_features(df, maxlen)[0]
    exact_lemmas = build_exact_lemma_features(df, maxlen)
    tf = build_term_frequency_features(df, maxlen)
    return doc_tags.tolist(), exact_lemmas.tolist(), tf.tolist()