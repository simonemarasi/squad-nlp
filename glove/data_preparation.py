from utils import is_whitespace, whitespace_tokenize
import pandas as pd
from constants import PUNCTUATION, EMPTY_TOKEN, MAX_CONTEXT_LEN, MAX_QUEST_LEN, PAD_TOKEN
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_examples(data, is_training = True):
  """
    Returns a Pandas DataFrame containing the samples of the data split passed.
    The DataFrame contains:
    - ID of the question
    - title of the passage
    - the list of words of the question
    - the list of words of the passage
    - the true answer
    - the start index of the answer in the passage tokens list
    - the end index of the answer in the passage tokens list
  """
  examples = []
  for entry in data:
    title = entry["title"]
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        #char to word offset indica l'indice in cui una parole termina nel context
        char_to_word_offset.append(len(doc_tokens) - 1)
      
      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        quest_tokens = whitespace_tokenize(question_text)
        start_position = None
        end_position = None
        orig_answer_text = None

        if is_training:
          if (len(qa["answers"]) != 1):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          
          answer = qa["answers"][0]
          orig_answer_text = answer["text"]
          answer_offset = answer["answer_start"]
          answer_length = len(orig_answer_text)
          start_position = char_to_word_offset[answer_offset]
          end_position = char_to_word_offset[answer_offset + answer_length - 1]
          # Only add answers where the text can be exactly recovered from the
          # document. If this CAN'T happen it's likely due to weird Unicode
          # stuff so we will just skip the example.
          #
          # Note that this means for training mode, every example is NOT
          # guaranteed to be preserved.
          actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
          cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
          if actual_text.find(cleaned_answer_text) == -1:
            tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                actual_text, cleaned_answer_text)
            continue
        else:
          start_position = -1
          end_position = -1
          orig_answer_text = ""

        example = { "qas_id": qas_id, 
                    "title": title, 
                    "quest_tokens": quest_tokens, 
                    "doc_tokens": doc_tokens, 
                    "orig_answer_text": orig_answer_text, 
                    "start_position": start_position, 
                    "end_position": end_position }
        examples.append(example)

  return pd.DataFrame(examples)


def preprocess_tokens(token_list):
  """
  Removes punctuation and symbols from tokens and returns a cleaned list of them
  """
  items = []
  for item in token_list:
    item = item.lower().translate(str.maketrans('', '', PUNCTUATION))
    if item == '':
        item = EMPTY_TOKEN
    items.append(item)
  return items    

def remove_too_long_samples(df):
    """
    Removes samples from the dataframe where number of tokens of either context or question is greater than thresholds
    """
    df[df['proc_doc_tokens'].map(len) <= MAX_CONTEXT_LEN]
    df[df['proc_quest_tokens'].map(len) <= MAX_QUEST_LEN]
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