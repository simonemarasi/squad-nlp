from common.utils import *
from common.functions import *
from glove.data_preparation import *
from common.additional_features_preparation import *
from glove.callback import *
from glove.glove_embedding import *
from glove.models import *
from glove.generators import *
from bidafLike.models import charCnnModel
from bidafLike.models import model
from config import *
from compute_answers import compute_predictions

def test_bidaf_like(filepath, outputdir):

    def preprocess_split(split):
        split = read_examples(split, False)
        split.sample(frac=1).reset_index(drop=True)
        split["proc_doc_tokens"] = split["doc_tokens"].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split["quest_tokens"].apply(preprocess_tokens)
        return split

    print("Loading Data")
    data = load_json_file(filepath)

    print("Preparing dataset")

    test = preprocess_split(data)
    print("Preparing embeddings")
    embedding_model = prepare_embedding_model(test, True)
    word2index, _ = build_embedding_indices(embedding_model)

    print("Processing tokens to be fed into model")
    
    X_test_quest, X_test_doc = embed_and_pad_sequences(test, word2index, embedding_model)
    X_test_ids = test["qas_id"].values
    X_test_doc_tokens = test["doc_tokens"].to_list()

    alphabet = list(ALPHABET)
    alphabet.extend(["<PAD>", "<UNK>"])
    index2char = list_to_dict(alphabet)
    char2index = {value: key for (key, value) in index2char.items()}

    char_embedding_matrix = build_char_embedding_matrix(embedding_model, index2char, char2index)
    CHARPAD = np.array([char2index['<PAD>'] for _ in range(MAX_WORD_LEN)])
    
    X_test_quest_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c]
                            for c in w] for w in s] for s in test["proc_quest_tokens"]]
    X_test_doc_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c]
                            for c in w] for w in s] for s in test["proc_doc_tokens"]]
    for i in range(len(X_test_quest_char)):
        X_test_quest_char[i] = pad_sequences(
            maxlen=MAX_WORD_LEN, sequences=X_test_quest_char[i], padding="post", truncating="post", value=char2index['<PAD>'])
    for i in range(len(X_test_doc_char)):
        X_test_doc_char[i] = pad_sequences(
            maxlen=MAX_WORD_LEN, sequences=X_test_doc_char[i], padding="post", truncating="post", value=char2index['<PAD>'])
    X_test_quest_char = pad_sequences(
        maxlen=MAX_QUEST_LEN, sequences=X_test_quest_char, padding="post", truncating="post", value=CHARPAD)
    X_test_doc_char = pad_sequences(
        maxlen=MAX_CONTEXT_LEN, sequences=X_test_doc_char, padding="post", truncating="post", value=CHARPAD)
    X_test = [X_test_quest, X_test_doc, X_test_quest_char, X_test_doc_char]

    embedding_matrix = np.zeros((len(word2index), EMBEDDING_DIMENSION))
    for word in word2index:
        embedding_matrix[word2index[word]] = embedding_model[word]

    print("Creating model:\n")

    doc_char_model = charCnnModel.buildCharCnnModel(input_shape=(MAX_CONTEXT_LEN, MAX_WORD_LEN),
                                                    embedding_size=EMBEDDING_DIMENSION,
                                                    conv_layers=CONV_LAYERS,
                                                    fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                                    dropout_p=0.1,
                                                    optimizer="adam",
                                                    loss="categorical_crossentropy",
                                                    num_classes=0,
                                                    char_embedding_matrix=char_embedding_matrix,
                                                    include_top=False,
                                                    train_embedding=True)
    doc_char_model.load_weights(CHAR_WEIGHTS_PATH)
    doc_char_model.trainable = False

    quest_char_model = charCnnModel.buildCharCnnModel(input_shape=(MAX_QUEST_LEN, MAX_WORD_LEN),
                                                      embedding_size=EMBEDDING_DIMENSION,
                                                      conv_layers=CONV_LAYERS,
                                                      fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                                      dropout_p=0.1,
                                                      optimizer="adam",
                                                      loss="categorical_crossentropy",
                                                      num_classes=0,
                                                      char_embedding_matrix=char_embedding_matrix,
                                                      include_top=False,
                                                      train_embedding=True)
    quest_char_model.load_weights(CHAR_WEIGHTS_PATH)
    quest_char_model.trainable = False

    bidafModel = model.buildBidafModel(embedding_matrix, doc_char_model, quest_char_model)
    weights_path = osp.join(BIDAF_WEIGHTS_PATH)

    print("\nLoading model weights:\n\n")
    MODEL_PATH = osp.join(weights_path, "weights.h5")
    bidafModel.load_weights(MODEL_PATH)
    compute_predictions(model, X_test, X_test_ids, X_test_doc_tokens, outputdir)

    return bidafModel