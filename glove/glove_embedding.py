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
    oov_words = [oov_words_quest, oov_words_doc]

    return list(set(oov_words))

def build_embedding_indices(embedding_model):
    """
    Build word2index and index2word dictionaries of the embedding model
    """
    index2word_list = embedding_model.index2word
    index2word  = list_to_dict(index2word_list)
    word2index = {value: key for (key, value) in index2word.items()}
    return word2index, index2word

