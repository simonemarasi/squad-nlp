from common.utils import is_whitespace, whitespace_tokenize
import pandas as pd
from config import PUNCTUATION, EMPTY_TOKEN
import re
import string
import collections
import json
import numpy as np
import os.path as osp


def read_examples(data, is_training=True):
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
                # char to word offset indica l'indice in cui una parole termina nel context
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                quest_tokens = whitespace_tokenize(question_text)
                start_position = None
                end_position = None
                orig_answer_text = None

                for a in qa["answers"]:
                    answer = a
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        continue
                    example = {"qas_id": qas_id,
                              "title": title,
                              "quest_tokens": quest_tokens,
                              "doc_tokens": doc_tokens,
                              "orig_answer_text": orig_answer_text,
                              "start_position": start_position,
                              "end_position": end_position}
                    examples.append(example)
                    if is_training:
                        # break the for loop getting only the first answer
                        break                    

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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_predictions(model, X, ids, tokens, outputdir):
    print("\nCompute predictions and saving to file")
    out = model.predict(X)
    start_idx = np.argmax(out[0], axis=-1).tolist()
    end_idx = np.argmax(out[1], axis=-1).tolist()
    result = {}
    for i in range(len(X[0])):
        idx = ids[i]
        start = start_idx[i]
        end = end_idx[i]
        pred = ' '.join(tokens[i][start:end+1])
        result[idx] = pred
    json_object = json.dumps(result)
    with open(osp.join(outputdir, "predictions.txt"), "w") as outfile:
        outfile.write(json_object)

def compute_bert_predictions(model, X, ids, tokens, lookup_list, input_ids, tokenizer, outputdir):
    out = model.predict(X)
    start_idx = np.argmax(out[0], axis=-1).tolist()
    end_idx = np.argmax(out[1], axis=-1).tolist()
    result = {}
    for i in range(len(ids)):
        idx = ids[i]
        start = start_idx[i]
        end = end_idx[i]
        try:          
            pred = " ".join(tokens[i][lookup_list[i][start]:(lookup_list[i][end] + 1)])
        except KeyError:
            pred = tokenizer.decode(input_ids[i][start:end+1])
        result[idx] = pred
    json_object = json.dumps(result)
    with open(osp.join(outputdir, "predictions.txt"), "w") as outfile:
        outfile.write(json_object)
