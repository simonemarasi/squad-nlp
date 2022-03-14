from common.constants import MAX_CONTEXT_LEN, MAX_QUEST_LEN, PAD_TOKEN
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np

def remove_not_valid_answer(df):
    """
    Removes samples from the dataframe that became invalid after removing the samples that were too long
    """
    df = df[df['start_position'] >= 0]
    df = df[df['start_position'] <= MAX_CONTEXT_LEN]

    df = df[df['end_position'] >= 0]
    df = df[df['end_position'] <= MAX_CONTEXT_LEN]
    return df

def remove_too_long_samples(df):
    """
    Removes samples from the dataframe where number of tokens of either context or question is greater than thresholds
    """
    df = df[df['proc_doc_tokens'].map(len) <= MAX_CONTEXT_LEN]
    df = df[df['proc_quest_tokens'].map(len) <= MAX_QUEST_LEN]
    return df

def embed_and_pad_sequences(df, word2index, embedding_model):
    """
    Converts tokens to GloVe sequences and pad them
    """
    X_quest = [[word2index['<UNK>'] if w not in embedding_model.vocab else word2index[w] for w in s] for s in df.proc_quest_tokens]
    X_doc = [[word2index['<UNK>'] if w not in embedding_model.vocab else word2index[w] for w in s] for s in df.proc_doc_tokens]

    X_quest = pad_sequences(maxlen=MAX_QUEST_LEN, sequences=X_quest, padding="post", truncating="post", value=word2index[PAD_TOKEN])
    X_doc = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_doc, padding="post", truncating="post", value=word2index[PAD_TOKEN])

    return X_quest, X_doc

def check_for_nan(list):
  for array in list:
    if np.isnan(np.sum(array)):
      return True
  return False
