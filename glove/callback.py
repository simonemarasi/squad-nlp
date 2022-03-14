from tensorflow.keras.callbacks import Callback
from common.functions import compute_f1, normalize_answer
import gc
import numpy as np

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
