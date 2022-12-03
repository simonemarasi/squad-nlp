import re
import string
import json
import collections

def evaluate_predictions(data_file, pred_file):
    with open(data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(pred_file) as f:
        preds = json.load(f)
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    out_eval = make_eval_dict(exact_raw, f1_raw)
    print(json.dumps(out_eval, indent=2))


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text']
                                for a in qa['answers'] if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred)
                                        for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred)
                                     for a in gold_answers)
    return exact_scores, f1_scores

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

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
