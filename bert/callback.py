from tensorflow.keras.callbacks import Callback
from common.functions import normalize_answer, compute_f1
import gc
import numpy as np

class ExactMatch(Callback):
    """
    Our Implementation
    """
    def __init__(self, x_eval, y_eval, doc_tokens_eval, bert_tokens_eval, y_text, eval_dict_map):
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.doc_tokens_eval = doc_tokens_eval
        self.bert_tokens_eval = bert_tokens_eval
        self.y_text = y_text
        self.eval_dict_map = eval_dict_map

    def on_epoch_end(self, epoch, tokenizer):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        f1_scores = []
        pred_start = np.argmax(pred_start, axis = 1)
        pred_end = np.argmax(pred_end, axis = 1)
        for start, end, tokens, bert_tokens, true_text, map in zip(pred_start, pred_end, self.doc_tokens_eval, self.bert_tokens_eval, self.y_text, self.eval_dict_map):
            try:          
                predicted_text = ' '.join(tokens[map[start]:(map[end] + 1)])
            except KeyError:
                predicted_text = tokenizer.decode(bert_tokens[start:end+1])
            normalized_pred_ans = normalize_answer(predicted_text)
            normalized_true_ans = normalize_answer(true_text)

            if normalized_pred_ans == normalized_true_ans:
                count += 1
            f1_scores.append(compute_f1(true_text, predicted_text))
        acc = 100.0 * (count / len(self.y_eval[0]))
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}%")   
        f1 = 100.0 * sum(f1_scores) / len(self.y_eval[0])
        print(f"F1-score={f1:.2f}%")
        gc.collect()
