from common.utils import *
from common.functions import *
from glove.data_preparation import *
from common.additional_features_preparation import *
from config import *
from glove.callback import *
from glove.glove_embedding import *
from glove.models import *
from common.layers import *
from glove.generators import *
from compute_answers import compute_predictions

def test_glove(filepath, model_choice, outputdir):

    print("Loading Data")
    data = load_json_file(filepath)

    print("Preparing dataset")

    def preprocess_split(split):
        split = read_examples(split, False)
        split.sample(frac=1).reset_index(drop=True)
        split["proc_doc_tokens"] = split["doc_tokens"].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split["quest_tokens"].apply(preprocess_tokens)
        return split

    test = preprocess_split(data)
    print("Preparing embeddings")
    embedding_model = prepare_embedding_model(test, True)

    word2index, _ = build_embedding_indices(embedding_model)
    pos_number = count_pos_tags()

    print("Creating model structure\n")
    if model_choice == "1":
        model = baseline_model(LEARNING_RATE, embedding_model)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "baseline")
    elif model_choice == "2":
        model = attention_model(LEARNING_RATE, embedding_model)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "attention")
    elif model_choice == "3":
        model = baseline_with_features(LEARNING_RATE, embedding_model, pos_number)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "features")
    elif model_choice == "4":
        model = attention_with_features(LEARNING_RATE, embedding_model, pos_number)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "attention-features")
    model.summary()

    X_test_quest, X_test_doc = embed_and_pad_sequences(test, word2index, embedding_model)
    X_test_ids = test["qas_id"].values
    X_test_doc_tokens = test["doc_tokens"].to_list()
    X_test = [X_test_quest, X_test_doc]

    if model_choice == "3" or model_choice == "4":
        # Computes additional features (POS, Exact Lemma, Term Frequency)
        print("Building additional features (it may take a while...)")
        X_test_doc_tags, pos_number = build_pos_features(test, MAX_CONTEXT_LEN)
        X_test_exact_lemma = build_exact_lemma_features(test, MAX_CONTEXT_LEN)
        X_test_tf = build_term_frequency_features(test, MAX_CONTEXT_LEN)
        X_test.extend([X_test_doc_tags, X_test_exact_lemma, X_test_tf])

    print("\nLoading model weights:\n\n")
    MODEL_PATH = osp.join(weights_path, "weights.h5")
    model.load_weights(MODEL_PATH)
    compute_predictions(model, X_test, X_test_ids, X_test_doc_tokens, outputdir)
    
    return model
