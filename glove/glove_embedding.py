import gensim
import gensim.downloader as gloader
import numpy as np
from constants import EMBEDDING_DIMENSION
from utils import list_to_dict

def load_embedding_model():
    """
    Loads a pre-trained GloVe embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """
    download_path = ""
    download_path = "glove-wiki-gigaword-{}".format(EMBEDDING_DIMENSION)
    emb_model = gloader.load(download_path)
    return emb_model

def handle_oov_words(embedding_model, oov_words):
    """
    Adds out-of-vocabulary words to embedding model
    """
    random_vectors = np.random.uniform(low=-4.0, high=4.0, size=(len(oov_words), EMBEDDING_DIMENSION))
    embedding_model.add(oov_words, random_vectors)
    return embedding_model

def get_oov_words_list(tokens, embedding_model):
    """
    Finds out-of-vocabulary words of the embedding model and returns a unique list of them
    """
    oov_words = [word for sentence in tokens for word in sentence if word not in embedding_model.vocab]
    return list(set(oov_words))

def build_embedding_indices(embedding_model):
    """
    Build word2index and index2word dictionaries of the embedding model
    """
    index2word_list = embedding_model.index2word
    index2word  = list_to_dict(index2word_list)
    word2index = {value: key for (key, value) in index2word.items()}
    return word2index, index2word

