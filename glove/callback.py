from tensorflow.keras.callbacks import Callback
import string
import re
import collections
import gc
import numpy as np

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
    if not s: return []
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


class ExactMatch(Callback):

    def __init__(self, x_eval, y_eval, doc_tokens_eval, y_text):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.doc_tokens_eval = doc_tokens_eval
        self.y_text = y_text

    def on_epoch_end(self, epoch, logs):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        f1_scores = []
        pred_start = np.argmax(pred_start, axis = -1)
        pred_end = np.argmax(pred_end, axis = -1)

        for start, end, tokens, true_text in zip(pred_start, pred_end, self.doc_tokens_eval, self.y_text):          
          predicted_text = " ".join(tokens[start:(end + 1)])
          normalized_pred_ans = normalize_answer(predicted_text)
          normalized_true_ans = normalize_answer(true_text)
          if normalized_pred_ans == normalized_true_ans:
                count += 1
          f1_scores.append(compute_f1(true_text, predicted_text))
          
        acc = 100 * (count / len(self.y_eval[0]))
        print(f"\nEpoch={epoch+1}, Exact Match score={acc:.2f}%")   
        f1 = 100.0 * sum(f1_scores) / len(self.y_eval[0])
        print(f"F1-score={f1:.2f}%\n")
        gc.collect()