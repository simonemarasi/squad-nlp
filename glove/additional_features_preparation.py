import nltk
from nltk.data import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from constants import PAD_POS, MAX_CONTEXT_LEN
import numpy as np

def build_pos_indices():
    """
    Build pos2index and index2pos dictionaries of the UPenn POS tags
    """
    nltk.download('tagsets')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    all_pos = [PAD_POS] + list(load("help/tagsets/upenn_tagset.pickle").keys())
    pos_to_idx = {pos: i for (i, pos) in enumerate(all_pos)}
    idx_to_pos = {i: pos for (i, pos) in enumerate(all_pos)}
    return idx_to_pos, pos_to_idx

def build_pos_features(df, pos_to_idx, idx_to_pos):
    """
    Return the list of the categorical representation of token POS tags padded
    """
    X_doc_tags = [nltk.pos_tag(s) for s in df.proc_doc_tokens]
    X_doc_tags = [[pos_to_idx[el[1]] for el in sequence] for sequence in X_doc_tags]
    X_doc_tags = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_doc_tags, padding="post", truncating="post", value=0)
    X_doc_tags = to_categorical(X_doc_tags, num_classes=len(idx_to_pos.keys()))

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
    tf = [tokens.count(ew)/l for ew in tokens]
    tf = np.array([tokens.count(ew)/l for ew in tokens])
    return np.reshape(tf, (l,-1))

def build_exact_lemma_features(df):
    X_train_exact_lemma = [exact_lemma(ques_tokens, con_tokens) for (ques_tokens, con_tokens) in zip(df.proc_quest_tokens, df.proc_doc_tokens)]
    return pad_sequences(X_train_exact_lemma, maxlen = MAX_CONTEXT_LEN, padding='post', truncating='post', value=np.array([0, 0]))

def build_term_frequency_features(df):
    X_train_tf = [term_frequency(tokens) for tokens in df.proc_doc_tokens]
    X_train_tf = pad_sequences(X_train_tf, maxlen = MAX_CONTEXT_LEN, padding='post', dtype='float64', truncating='post', value=0.0)