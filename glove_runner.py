from common.utils import *
from common.functions import *
from glove.data_preparation import *
from common.additional_features_preparation import *
from config import *
from glove.callback import *
from glove.glove_embedding import *
from glove.models import *
from glove.layers import *
from glove.generators import *
import os
from datetime import datetime

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3", "4"]:
            print("Sorry, your choice must be between the four allowed")
            continue
        else:
            break
    return value

def glove_runner(filepath, outputdir=GLOVE_WEIGHTS_PATH):

    print("######################")
    print("#### GLOVE RUNNER ####")
    print("######################")
    print("\nModels available:\n")
    print("1) Baseline")
    print("2) Baseline with attention")
    print("3) Baseline with features")
    print("4) Baseline with attention and features")
    model_choice = get_model_input("\nPlease type the model number to run with the current configuration: ")

    print("Loading Data")
    data = load_json_file(filepath)

    train = data[:VAL_SPLIT_INDEX]
    eval = data[VAL_SPLIT_INDEX:]

    print("Preparing dataset")

    def preprocess_split(split):
        split = read_examples(split)
        split.sample(frac=1).reset_index(drop=True)
        split["proc_doc_tokens"] = split['doc_tokens'].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split['quest_tokens'].apply(preprocess_tokens)
        split = remove_outliers(split)
        split = remove_not_valid_answer(split)
        return split

    train = preprocess_split(train)
    eval = preprocess_split(eval)

    print("Preparing embedding")
    embedding_model = load_embedding_model()
    embedding_model = add_oov_words(train, embedding_model)
    word2index, index2word = build_embedding_indices(embedding_model)

    print("Processing tokens to be fed into model")
    X_train_quest, X_train_doc = embed_and_pad_sequences(train, word2index, embedding_model)
    X_val_quest, X_val_doc = embed_and_pad_sequences(eval, word2index, embedding_model)

    y_train_start = train["start_position"].to_numpy()
    y_train_end = train["end_position"].to_numpy()
    y_val_start = eval["start_position"].to_numpy()
    y_val_end = eval["end_position"].to_numpy()
    val_doc_tokens = eval["doc_tokens"].to_numpy()
    val_answer_text = eval["orig_answer_text"].to_numpy()

    print("Building additional features (it may take a while...)")
    X_train_doc_tags, pos_number = build_pos_features(train, MAX_CONTEXT_LEN)
    X_train_exact_lemma = build_exact_lemma_features(train, MAX_CONTEXT_LEN)
    X_train_tf = build_term_frequency_features(train, MAX_CONTEXT_LEN)

    X_eval_doc_tags, pos_number = build_pos_features(eval, MAX_CONTEXT_LEN)
    X_eval_exact_lemma = build_exact_lemma_features(eval, MAX_CONTEXT_LEN)
    X_eval_tf = build_term_frequency_features(eval, MAX_CONTEXT_LEN)

    X_train = [X_train_quest, X_train_doc, X_train_doc_tags, X_train_exact_lemma, X_train_tf]
    y_train = [y_train_start, y_train_end]
    X_val = [X_val_quest, X_val_doc, X_eval_doc_tags, X_eval_exact_lemma, X_eval_tf]
    y_val = [y_val_start, y_val_end]

    print("Freeing up memory, cleaning unused variables")
    del train
    del eval
    gc.collect()

    print("Fitting data to generators")
    TRAIN_LEN = X_train[0].shape[0]
    VAL_LEN = X_val[0].shape[0]

    train_generator = features_data_generator(X_train, y_train, BATCH_SIZE)
    val_generator = features_data_generator(X_val, y_val, BATCH_SIZE)

    print("Creating model:\n")
    if model_choice == "1":
        model = baseline_model(LEARNING_RATE, embedding_model)
    elif model_choice == "2":
        model = baseline_with_features(LEARNING_RATE, embedding_model, pos_number)
    elif model_choice == "3":
        model = attention_model(LEARNING_RATE, embedding_model)
    elif model_choice == "4":
        model = attention_with_features(LEARNING_RATE, embedding_model, pos_number)

    model.summary()

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    exact_match_callback = ExactMatch(X_val, y_val, val_doc_tokens, val_answer_text)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    print("\nTrain start:\n\n")
    history = model.fit(
        train_generator,
        validation_data = val_generator,
        steps_per_epoch = TRAIN_LEN / BATCH_SIZE,
        validation_steps = VAL_LEN / BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[exact_match_callback, es],
        workers = WORKERS
    )

    print("### SAVING MODEL ###")
    model.save_weights(os.path.join(outputdir,'weights-{}.h5'.format(now)))
    print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
    print("")
    print("### SAVING HISTORY ###")
    df_hist = pd.DataFrame.from_dict(history.history)
    df_hist.to_csv(os.path.join(outputdir,'history-{}.csv'.format(now)), mode='w', header=True)
