import numpy as np
import json
import os.path as osp

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
            pred = ' '.join(tokens[i][lookup_list[i][start]:(lookup_list[i][end] + 1)])
        except KeyError:
            pred = tokenizer.decode(input_ids[i][start:end+1])
        result[idx] = pred
    json_object = json.dumps(result)
    with open(osp.join(outputdir, "predictions.txt"), "w") as outfile:
        outfile.write(json_object)
