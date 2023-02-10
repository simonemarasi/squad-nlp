from tensorflow.keras.callbacks import Callback
from common.functions import compute_f1, normalize_answer
from config import BATCH_SIZE
import gc
import numpy as np

class ExactMatch(Callback):
    """ Callback used during training to compute and monitor the metrics each epoch"""
    def __init__(self, x_eval, y_eval, doc_tokens_eval, y_text):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.doc_tokens_eval = doc_tokens_eval
        self.y_text = y_text

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval, batch_size=BATCH_SIZE)
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
        f1 = 100.0 * sum(f1_scores) / len(self.y_eval[0])
        print(f"\nEpoch={epoch+1}, Exact Match score={acc:.2f}%, F1-score={f1:.2f}%\n")
        gc.collect()

class LearningRateReducer(Callback):
  """ Callback used during training to reduce the learning rate by a fixed factor"""
  def __init__(self, downscale_factor):
    self.downscale_factor = downscale_factor

  def on_epoch_end(self, epoch, logs=None):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * self.downscale_factor
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}\n".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)
    