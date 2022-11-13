import os
import os.path as osp
from config import BERT_SAVE_DIR, BERT_MODEL, PAD_POS, TRANSL_DICT, BERT_MAX_LEN
import numpy as np
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from common.additional_features_preparation import build_pos_indices

def save_bert_tokenizer():
    """
    Save the slow pretrained tokenizer
    """
    slow_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    save_path = BERT_SAVE_DIR
    if not osp.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

def load_bert_tokenizer():
    """
    Load the BERT tokenizer
    """
    save_bert_tokenizer()
    return BertWordPieceTokenizer(osp.join(BERT_SAVE_DIR, "vocab.txt"), lowercase=True)

def bert_tokenization(token_list, tokenizer):
    """
    Encode token by token the list of tokens using the tokenizer
    """
    return [tokenizer.encode(token_list[i], add_special_tokens=False) for i in range(len(token_list))]

def unpack_dataframe(df, with_features=True):
  tot_input_ids = []
  tot_token_type_ids = []
  tot_attention_mask = []
  if with_features:
    tot_pos_tags = []
    tot_exact_lemmas = []
    tot_term_frequency = []

  tot_bert_start = []
  tot_bert_end = []

  tot_doc_tokens =[]
  tot_orig_answer_text = []
  
  for _, row in tqdm(df.iterrows(), total=len(df)):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    if with_features:
      pos_tags = []
      exact_lemmas = []
      term_frequency = []
    orig_doc_tokens = row.doc_tokens
    orig_answer_text = row.orig_answer_text

    bert_start = 0
    bert_end = 0
    bert_token_counter = 0

    if with_features:
      tokens = zip(row.bert_tokenized_doc_tokens, row.pos_tag, row.exact_lemma, row.tf)
    else:
      tokens = row.bert_tokenized_doc_tokens

    for id, doc_tokens in enumerate(tokens):  

      if with_features:
        doc_tokens, pos, el, tf = doc_tokens

      if id == 0:
        input_ids.append(101)
        token_type_ids.append(0)
        attention_mask.append(1)
        if with_features:
          pos_tags.append(PAD_POS)
          exact_lemmas.append(np.array([0,0]))
          term_frequency.append(np.array([0.0]))
        bert_token_counter += 1

      if id == row.start_position:
        bert_start = bert_token_counter

      for (int_id, tok) in zip(doc_tokens.ids, doc_tokens.tokens):
        bert_token_counter += 1
        input_ids.append(int_id)
        token_type_ids.append(0)
        attention_mask.append(1)

        if with_features:
          if tok in TRANSL_DICT.keys():
            pos_tags.append(TRANSL_DICT[tok])
            exact_lemmas.append(np.array([0,0]))
            term_frequency.append(np.array([0.0]))
          else:
            pos_tags.append(pos[1])
            exact_lemmas.append(el)
            term_frequency.append(tf)
      
      if id == row.end_position:
        bert_end = bert_token_counter-1

    for id, quest_tokens in enumerate(row.bert_tokenized_quest_tokens):
      if id == 0:
        input_ids.append(102)
        token_type_ids.append(0)
        attention_mask.append(1)
        if with_features:
          pos_tags.append(PAD_POS)
          exact_lemmas.append(np.array([0,0]))
          term_frequency.append(np.array([0.0]))

      for (int_id, tok) in zip(quest_tokens.ids, quest_tokens.tokens):
        input_ids.append(int_id)
        token_type_ids.append(1)
        attention_mask.append(1)
        if with_features:
          pos_tags.append(PAD_POS)
          exact_lemmas.append(np.array([0,0]))
          term_frequency.append(np.array([0.0]))
      
    input_ids.append(102)
    token_type_ids.append(1)
    attention_mask.append(1)
    if with_features:
      pos_tags.append(PAD_POS)
      exact_lemmas.append(np.array([0,0]))
      term_frequency.append(np.array([0.0]))
      
    tot_doc_tokens.append(orig_doc_tokens)
    tot_orig_answer_text.append(orig_answer_text)
    tot_input_ids.append(input_ids)
    tot_token_type_ids.append(token_type_ids)
    tot_attention_mask.append(attention_mask)

    if with_features:
      tot_pos_tags.append(pos_tags)
      tot_exact_lemmas.append(exact_lemmas)
      tot_term_frequency.append(term_frequency)

    tot_bert_start.append(bert_start)
    tot_bert_end.append(bert_end)

  tot_bert_start = np.array(tot_bert_start)
  tot_bert_end = np.array(tot_bert_end)
  if with_features:
    return tot_input_ids, tot_token_type_ids, tot_attention_mask, tot_pos_tags, tot_exact_lemmas, tot_term_frequency, tot_bert_start, tot_bert_end, tot_doc_tokens, tot_orig_answer_text
  else:
    return tot_input_ids, tot_token_type_ids, tot_attention_mask, tot_bert_start, tot_bert_end, tot_doc_tokens, tot_orig_answer_text

def pad_inputs(input_ids, token_type_ids, attention_mask):
    input_ids = pad_sequences(input_ids, padding="post", truncating="post", maxlen=BERT_MAX_LEN)
    token_type_ids = pad_sequences(token_type_ids, padding="post", truncating="post", maxlen=BERT_MAX_LEN)
    attention_mask = pad_sequences(attention_mask, padding="post", truncating="post", maxlen=BERT_MAX_LEN)
    return input_ids, token_type_ids, attention_mask

def pad_additonal_features(pos_tags, exact_lemma, term_freq):
    pos_to_idx, _ = build_pos_indices()
    pos_tags = [[pos_to_idx[el] for el in sequence] for sequence in pos_tags]
    pos_tags = pad_sequences(pos_tags, padding='post', truncating='post', maxlen=BERT_MAX_LEN, value = 0)
    pos_tags = to_categorical(pos_tags, num_classes=len(pos_to_idx.keys()))
    exact_lemma = pad_sequences(exact_lemma, padding='post', truncating='post', maxlen=BERT_MAX_LEN, value=np.array([0, 0]))
    term_freq = pad_sequences(term_freq, padding='post', truncating='post', maxlen=BERT_MAX_LEN, value = 0.0)
    return pos_tags, exact_lemma, term_freq

def compute_lookups(df):
  """
  Build a lookup table between the tokenized token and its original position.
  This is due the BERT lemmatization that could mess up the final start and end answer indices.
  """
  lookup_list = []
  for _, row in tqdm(df.iterrows(), total=len(df)): 
    lookup_dict = {}
    id = 0
    j = 1
    for el in row.bert_tokenized_doc_tokens:
      for _ in range(len(el)):
        lookup_dict[j] = id
        j += 1
      id += 1
    lookup_list.append(lookup_dict)
  return lookup_list
  