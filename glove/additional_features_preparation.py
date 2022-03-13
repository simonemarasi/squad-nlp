import nltk
from nltk.data import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from glove.constants import PAD_POS, MAX_CONTEXT_LEN
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

def build_pos_features(df):
    """
    Return the list of the categorical representation of token POS tags padded
    """
    pos_to_idx, idx_to_pos = build_pos_indices()
    doc_tags = [nltk.pos_tag(s) for s in df.proc_doc_tokens]
    doc_tags = [[pos_to_idx[el[1]] for el in sequence] for sequence in doc_tags]
    doc_tags = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=doc_tags, padding="post", truncating="post", value=0)
    doc_tags = to_categorical(doc_tags, num_classes=len(idx_to_pos.keys()))
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
    tf = [tokens.count(ew)/l for ew in tokens]
    tf = np.array([tokens.count(ew)/l for ew in tokens])
    return np.reshape(tf, (l,-1))

def build_exact_lemma_features(df):
    lemma_array = [exact_lemma(quest_tokens, doc_tokens) for (quest_tokens, doc_tokens) in zip(df.proc_quest_tokens, df.proc_doc_tokens)]  
    return pad_sequences(lemma_array, maxlen = MAX_CONTEXT_LEN, padding='post', truncating='post', value=np.array([0, 0]))

def build_term_frequency_features(df):
    tf = [term_frequency(tokens) for tokens in df.proc_doc_tokens]
    return pad_sequences(tf, maxlen = MAX_CONTEXT_LEN, padding='post', dtype='float64', truncating='post', value=0.0)