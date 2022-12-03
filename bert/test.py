from common.utils import *
from common.functions import *
from bert.data_preparation import *
from common.additional_features_preparation import *
from config import *
from bert.callback import *
from bert.models import *
from bert.generators import *
from sklearn.utils import shuffle
from compute_answers import compute_bert_predictions

def test_bert(filepath, model_choice, outputdir):
    print("Loading Data")
    data = load_json_file(filepath)

    print("Preparing dataset")

    def preprocess_split(split):
        split = read_examples(split, False)
        split = shuffle(split)
        split["proc_doc_tokens"] = split["doc_tokens"].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split["quest_tokens"].apply(preprocess_tokens)
        split["bert_tokenized_doc_tokens"] = split["doc_tokens"].apply(bert_tokenization, tokenizer=tokenizer)
        split["bert_tokenized_quest_tokens"] = split["quest_tokens"].apply(bert_tokenization, tokenizer=tokenizer)
        if (model_choice == "3"):
            split["pos_tag"], split["exact_lemma"], split["tf"] = get_additional_features(split, BERT_MAX_LEN)
        return split

    tokenizer = load_bert_tokenizer()

    test = preprocess_split(data)
    if (model_choice == "3"):
        print("Preparing additional features")
        X_test_input_ids, X_test_token_type_ids, X_test_attention_mask, X_test_pos_tags, X_test_exact_lemma, X_test_tf, _, _, test_doc_tokens, _ = unpack_dataframe(test)
    else:
        X_test_input_ids, X_test_token_type_ids, X_test_attention_mask, _, _, test_doc_tokens, _ = unpack_dataframe(test, with_features=False)

    test_lookup_list = compute_lookups(test)
    X_test_qas_id = test["qas_id"].values.tolist()

    X_test_input_ids, X_test_token_type_ids, X_test_attention_mask = pad_inputs(X_test_input_ids, X_test_token_type_ids, X_test_attention_mask)
    X_test = [X_test_input_ids, X_test_token_type_ids, X_test_attention_mask]

    if (model_choice == "3"):
        X_test_pos_tags, X_test_exact_lemma, X_test_tf = pad_additonal_features(X_test_pos_tags, X_test_exact_lemma, X_test_tf)
        X_test.extend([X_test_pos_tags, X_test_exact_lemma, X_test_tf])

    print("Creating model:\n")
    if model_choice == "1":
        model = baseline_model(LEARNING_RATE)
        weights_path = osp.join(BERT_WEIGHTS_PATH, "baseline")
    elif model_choice == "2":
        model = baseline_with_rnn(LEARNING_RATE)
        weights_path = osp.join(BERT_WEIGHTS_PATH, "rnn")
    elif model_choice == "3":
        model = features_with_rnn(LEARNING_RATE)
        weights_path = osp.join(BERT_WEIGHTS_PATH, "rnn-features")

    model.summary()

    print("\nLoading model weights\n\n")
    MODEL_PATH = osp.join(weights_path, "weights.h5")
    model.load_weights(MODEL_PATH)
    print("\nComputing predictions and save into a file\n\n")
    compute_bert_predictions(model, X_test, X_test_qas_id, test_doc_tokens, test_lookup_list, X_test_input_ids, tokenizer, outputdir)

    return model