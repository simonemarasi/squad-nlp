
import numpy as np
import json

def compute_bert_answers(model, X, qas_id, doc_tokens, bert_tokens, tokenizer, map_fn):
    pred = model.predict(X)

    start_idx = np.argmax(pred[0], axis=-1).tolist()
    end_idx = np.argmax(pred[1], axis=-1).tolist()

    result = {}
    for i in range(len(qas_id)):
        idx = qas_id[i]
        start = start_idx[i]
        end = end_idx[i]
        try:          
            pred = " ".join(doc_tokens[i][map_fn[i][start]:(map_fn[i][end] + 1)])
        except KeyError:
            pred = tokenizer.decode(bert_tokens[i][start:end+1])
        result[idx] = pred

    json_object = json.dumps(result)

    with open("predictions.json", "w") as f:
        f.write(json_object)

def compute_glove_answers(model, X, ids, tokens):
    #X_val = [X_val_quest, X_val_doc]
    #X_val = [X_val_quest, X_val_doc, X_val_doc_tags, X_val_exact_lemma, X_val_tf]
    pred = model.predict(X)

    start_idx = np.argmax(pred[0], axis=-1).tolist()
    end_idx = np.argmax(pred[1], axis=-1).tolist()

    result = {}
    for i in range(len(X[0])):
        idx = ids[i]
        start = start_idx[i]
        end = end_idx[i]
        pred = ' '.join(tokens[i][start:end+1])
        result[idx] = pred

    json_object = json.dumps(result)

    with open("predictions.json", "w") as f:
        f.write(json_object)