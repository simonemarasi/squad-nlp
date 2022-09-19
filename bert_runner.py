from common.utils import *
from common.functions import *
from bert.data_preparation import *
from common.additional_features_preparation import *
from config import *
from bert.callback import *
from bert.models import *
from bert.generators import *
import os
from datetime import datetime
from sklearn.utils import shuffle

print("#####################")
print("#### BERT RUNNER ####")
print("#####################")
print("\nModels available:\n")
print("1) Baseline")
print("2) Baseline with RNN")
print("3) Baseline with RNN and features")

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3"]:
            print("Sorry, your choice must be between the three allowed")
            continue
        else:
            break
    return value

def bert_runner(filepath, outputdir):

    model_choice = get_model_input("\nPlease type the model number to run with the current configuration: ")

    print("Loading Data")
    data = load_json_file(filepath)

    train = data[:VAL_SPLIT_INDEX]
    eval = data[VAL_SPLIT_INDEX:]

    tokenizer = load_bert_tokenizer()

    def preprocess_split(split):
        split = read_examples(split)
        split = shuffle(split)
        split["proc_doc_tokens"] = split['doc_tokens'].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split['quest_tokens'].apply(preprocess_tokens)
        split["bert_tokenized_doc_tokens"] = split['doc_tokens'].apply(bert_tokenization, tokenizer=tokenizer)
        split["bert_tokenized_quest_tokens"] = split['quest_tokens'].apply(bert_tokenization, tokenizer=tokenizer)
        return split

    def get_additional_features(split):
        doc_tags = build_pos_features(split, BERT_MAX_LEN)[0]
        exact_lemma = build_exact_lemma_features(split, BERT_MAX_LEN)
        tf = build_term_frequency_features(split, BERT_MAX_LEN)
        return doc_tags, exact_lemma, tf

    print("Preparing dataset")
    train = preprocess_split(train)
    eval = preprocess_split(eval)

    if (model_choice == "3"):
        print("Preparing additional features")
        X_train_doc_tags, X_train_exact_lemma, X_train_tf = get_additional_features(train)
        X_eval_doc_tags, X_eval_exact_lemma, X_eval_tf = get_additional_features(eval)

    if (model_choice == "3"):
        X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, X_train_pos_tags, X_train_exact_lemmas, X_train_term_frequency, y_train_start, y_train_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train)
        X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, X_eval_pos_tags, X_eval_exact_lemmas, X_eval_term_frequency, y_eval_start, y_eval_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval)
    else:
        X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, y_train_start, y_train_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train, with_features=False)
        X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, y_eval_start, y_eval_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval, with_features=False)

    def pad_inputs(input_ids, token_type_ids, attention_mask):
        input_ids = pad_sequences(X_train_input_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
        token_type_ids = pad_sequences(X_train_token_type_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
        attention_mask = pad_sequences(X_train_attention_mask, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
        return input_ids, token_type_ids, attention_mask

    #train_lookup_list = compute_lookups(train)
    eval_lookup_list = compute_lookups(eval)

    X_train_input_ids, X_train_token_type_ids, X_train_attention_mask = pad_inputs(X_train_input_ids, X_train_token_type_ids, X_train_attention_mask)
    X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask = pad_inputs(X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask)

    if (model_choice == "3"):
        X_train = [X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, X_train_doc_tags, X_train_exact_lemma, X_train_tf]
        X_val = [X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, X_eval_doc_tags, X_eval_exact_lemma, X_eval_tf]
    else:
        X_train = [X_train_input_ids, X_train_token_type_ids, X_train_attention_mask]
        X_val = [X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask]
    y_train = [y_train_start, y_train_end]
    y_val = [y_eval_start, y_eval_end]

    print("Fitting data to generators")
    TRAIN_LEN = X_train[0].shape[0]
    VAL_LEN = X_val[0].shape[0]

    train_generator = features_data_generator(X_train, y_train, BATCH_SIZE)
    val_generator = features_data_generator(X_val, y_val, BATCH_SIZE)

    print("Creating model:\n")
    if model_choice == "1":
        model = baseline_model(LEARNING_RATE)
    elif model_choice == "2":
        model = baseline_with_rnn(LEARNING_RATE)
    elif model_choice == "3":
        model = features_with_rnn(LEARNING_RATE)

    model.summary()

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    exact_match_callback = ExactMatch(X_val, y_val, eval_doc_tokens, X_eval_input_ids, eval_orig_answer_text, eval_lookup_list)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data = val_generator,
        steps_per_epoch = TRAIN_LEN / BATCH_SIZE,
        validation_steps = VAL_LEN / BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[exact_match_callback, es],
        workers=WORKERS
    )

    print("### SAVING MODEL ###")
    model.save_weights(os.path.join(outputdir,'weights-{}.h5'.format(now)))
    print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
    print("")
    print("### SAVING HISTORY ###")
    df_hist = pd.DataFrame.from_dict(history.history)
    df_hist.to_csv(os.path.join(outputdir,'history-{}.csv'.format(now)), mode='w', header=True)
