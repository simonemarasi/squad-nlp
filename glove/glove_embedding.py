import gensim.downloader as gloader
import numpy as np
from config import EMBEDDING_DIMENSION, EMBEDDING_PATH
from common.utils import list_to_dict
import pickle

def prepare_embedding_model(df, load_embedding):
    """
    Loads a pre-trained GloVe embedding model via gensim library or via pickle pre-generated file
    """
    if load_embedding:
        with open(EMBEDDING_PATH, 'rb') as f:
            emb_model = pickle.load(f)
    else:
        download_path = "glove-wiki-gigaword-{}".format(EMBEDDING_DIMENSION)
        emb_model = gloader.load(download_path)
        emb_model = add_oov_words(df, emb_model)
    return emb_model

def build_char_embedding_matrix(embedding_model, index2char, char2index):
    char_embedding_matrix = np.zeros((len(index2char), EMBEDDING_DIMENSION))
    for index in index2char:
        if index == char2index["<PAD>"]:
            np.zeros(shape=(1, EMBEDDING_DIMENSION))
        elif index == char2index["<UNK>"]:
            np.random.uniform(low=-4.0, high=4.0, size=(1, EMBEDDING_DIMENSION))
        else:
            char_embedding_matrix[index] = embedding_model[index2char[index]]
    return char_embedding_matrix

def add_oov_words(df, embedding_model):
    """
    Adds out-of-vocabulary words to embedding model
    """
    oov_words = get_oov_words_list(df, embedding_model)
    random_vectors = np.random.uniform(low=-4.0, high=4.0, size=(len(oov_words), EMBEDDING_DIMENSION))
    embedding_model.add(oov_words, random_vectors)

    embedding_model['<PAD>'] = np.zeros(shape=(1, EMBEDDING_DIMENSION))
    embedding_model['<UNK>'] = np.random.uniform(low=-4.0, high=4.0, size=(1, EMBEDDING_DIMENSION))

    return embedding_model

def get_oov_words_list(df, embedding_model):
    """
    Finds out-of-vocabulary words of the embedding model and returns a unique list of them
    """
    oov_words_doc = [word for sentence in df.proc_doc_tokens for word in sentence if word not in embedding_model.vocab]
    oov_words_quest = [word for sentence in df.proc_quest_tokens for word in sentence if word not in embedding_model.vocab]
    oov_words = oov_words_quest + oov_words_doc

    return list(set(oov_words))

def build_embedding_indices(embedding_model):
    """
    Build word2index and index2word dictionaries of the embedding model
    """
    index2word_list = embedding_model.index2word
    index2word  = list_to_dict(index2word_list)
    word2index = {value: key for (key, value) in index2word.items()}
    return word2index, index2word
