from common.utils import *
from common.functions import *
from glove.data_preparation import *
from common.additional_features_preparation import *
from config import *
from glove.callbacks import *
from glove.glove_embedding import *
from glove.model.models import *
from glove.generators import *
import os
from tensorflow.keras.callbacks import EarlyStopping
from glove.model import charCnnModel

def train_glove(filepath, load_embedding, model_choice):
    print("Loading Data")
    data = load_json_file(filepath)

    print("Preparing dataset")

    def preprocess_split(split):
        split = read_examples(split, True)
        split.sample(frac=1).reset_index(drop=True)
        split["proc_doc_tokens"] = split["doc_tokens"].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split["quest_tokens"].apply(
            preprocess_tokens)
        split = remove_outliers(split)
        split = remove_not_valid_answer(split)
        return split

    train = data[:VAL_SPLIT_INDEX]
    eval = data[VAL_SPLIT_INDEX:]
    train = preprocess_split(train)
    eval = preprocess_split(eval)
    print("Preparing embeddings")
    embedding_model = prepare_embedding_model(train, load_embedding)
    word2index, _ = build_embedding_indices(embedding_model)
    embedding_matrix = np.zeros((len(word2index), EMBEDDING_DIMENSION))
    for word in word2index:
        embedding_matrix[word2index[word]] = embedding_model[word]

    print("Processing tokens to be fed into model")
    X_train_quest, X_train_doc = embed_and_pad_sequences(train, word2index, embedding_model)
    X_val_quest, X_val_doc = embed_and_pad_sequences(eval, word2index, embedding_model)

    y_train_start = train["start_position"].to_numpy()
    y_train_end = train["end_position"].to_numpy()
    y_val_start = eval["start_position"].to_numpy()
    y_val_end = eval["end_position"].to_numpy()
    val_doc_tokens = eval["doc_tokens"].to_numpy()
    val_answer_text = eval["orig_answer_text"].to_numpy()

    X_train = [X_train_quest, X_train_doc]
    X_val = [X_val_quest, X_val_doc]
    y_train = [y_train_start, y_train_end]
    y_val = [y_val_start, y_val_end]

    if model_choice == "3":
        # Computes additional features (POS, Exact Lemma, Term Frequency)
        print("Building additional features (it may take a while...)")
        X_train_exact_lemma = build_exact_lemma_features(
            train, MAX_CONTEXT_LEN)
        X_train_tf = build_term_frequency_features(train, MAX_CONTEXT_LEN)
        X_train.extend([X_train_exact_lemma, X_train_tf])

        X_eval_exact_lemma = build_exact_lemma_features(eval, MAX_CONTEXT_LEN)
        X_eval_tf = build_term_frequency_features(eval, MAX_CONTEXT_LEN)
        X_val.extend([X_eval_exact_lemma, X_eval_tf])

    if model_choice == "4":
        alphabet = list(ALPHABET)
        alphabet.extend([PAD_TOKEN, UNK_TOKEN])
        index2char = list_to_dict(alphabet)
        char2index = {value: key for (key, value) in index2char.items()}

        char_embedding_matrix = build_char_embedding_matrix(embedding_model, index2char, char2index)
        CHARPAD = np.array([char2index[PAD_TOKEN] for _ in range(MAX_WORD_LEN)])
        
        X_train_quest_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in train["proc_quest_tokens"]]
        X_train_doc_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in train["proc_doc_tokens"]]
        X_val_quest_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in eval["proc_quest_tokens"]]
        X_val_doc_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in eval["proc_doc_tokens"]]

        for i in range(len(X_train_quest_char)):
            X_train_quest_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_train_quest_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])
        for i in range(len(X_train_doc_char)):
            X_train_doc_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_train_doc_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])
        for i in range(len(X_val_quest_char)):
            X_val_quest_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_val_quest_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])
        for i in range(len(X_val_doc_char)):
            X_val_doc_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_val_doc_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])

        X_train_quest_char = pad_sequences(maxlen=MAX_QUEST_LEN, sequences=X_train_quest_char, padding="post", truncating="post", value=CHARPAD)
        X_train_doc_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_train_doc_char, padding="post", truncating="post", value=CHARPAD)
        X_val_quest_char = pad_sequences(maxlen=MAX_QUEST_LEN, sequences=X_val_quest_char, padding="post", truncating="post", value=CHARPAD)
        X_val_doc_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_val_doc_char, padding="post", truncating="post", value=CHARPAD)

        X_train = [X_train_quest, X_train_doc, X_train_quest_char, X_train_doc_char]
        y_train = [y_train_start, y_train_end]
        X_val = [X_val_quest, X_val_doc, X_val_quest_char, X_val_doc_char]
        y_val = [y_val_start, y_val_end]

        print("Creating CNN model:\n")

        doc_char_model = charCnnModel.buildCharCnnModel(input_shape=(MAX_CONTEXT_LEN, MAX_WORD_LEN),
                                                        embedding_size=EMBEDDING_DIMENSION,
                                                        conv_layers=CONV_LAYERS,
                                                        fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                                        char_embedding_matrix=char_embedding_matrix,
                                                        include_top=False)
        doc_char_model.load_weights(CHAR_WEIGHTS_PATH)
        doc_char_model.trainable = False

        quest_char_model = charCnnModel.buildCharCnnModel(input_shape=(MAX_QUEST_LEN, MAX_WORD_LEN),
                                                        embedding_size=EMBEDDING_DIMENSION,
                                                        conv_layers=CONV_LAYERS,
                                                        fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                                        char_embedding_matrix=char_embedding_matrix,
                                                        include_top=False)
        quest_char_model.load_weights(CHAR_WEIGHTS_PATH)
        quest_char_model.trainable = False

    print("Freeing up memory, cleaning unused variables")
    del train
    del eval
    gc.collect()

    print("Fitting data to generators")
    TRAIN_LEN = X_train[0].shape[0]
    VAL_LEN = X_val[0].shape[0]

    if (model_choice == "3"):
        train_generator = features_data_generator(X_train, y_train, BATCH_SIZE)
        val_generator = features_data_generator(X_val, y_val, BATCH_SIZE)
    else:
        train_generator = baseline_data_generator(X_train, y_train, BATCH_SIZE)
        val_generator = baseline_data_generator(X_val, y_val, BATCH_SIZE)

    exact_match_callback = ExactMatch(
        X_val, y_val, val_doc_tokens, val_answer_text)
    lrr = LearningRateReducer(0.8)
    es = EarlyStopping(monitor='val_loss', patience=3)

    print("Creating model structure\n")
    if model_choice == "1":
        model = build_model(embedding_matrix, LEARNING_RATE)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "baseline")
    elif model_choice == "2":
        model = build_model(embedding_matrix, LEARNING_RATE, attention=True)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "attention")
    elif model_choice == "3":
        model = build_model(embedding_matrix, LEARNING_RATE, features=True)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "features")
    elif model_choice == "4":
        model = build_model(embedding_matrix, LEARNING_RATE, char_embedding=True, doc_char_model=doc_char_model, quest_char_model=quest_char_model)
        weights_path = osp.join(GLOVE_WEIGHTS_PATH, "char")
    model.summary()

    print("\nTrain start:\n\n")
    model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=TRAIN_LEN / BATCH_SIZE,
        validation_steps=VAL_LEN / BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[exact_match_callback, es, lrr],
        shuffle=True,
        workers=WORKERS)
    print("### SAVING MODEL ###")
    model.save_weights(os.path.join(weights_path, 'weights.h5'))
    print("Weights saved to: " + weights_path +
          "/weights.h5 inside the model directory")

    return model
