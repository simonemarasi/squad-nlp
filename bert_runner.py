from common.utils import *
from common.functions import *
from bert.data_preparation import *
from common.additional_features_preparation import *
from config import *
from bert.callback import *
from bert.models import *
from bert.generators import *
import os
from sklearn.utils import shuffle

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3"]:
            print("Sorry, your choice must be between the three allowed")
            continue
        else:
            break
    return value

def bert_runner(filepath, outputdir=BERT_WEIGHTS_PATH, mode="test"):

    print("#####################")
    print("#### BERT RUNNER ####")
    print("#####################")
    print("\nModels available:\n")
    print("1) Baseline")
    print("2) Baseline with RNN")
    print("3) Baseline with RNN and features")

    model_choice = get_model_input("\nPlease type the model number to run with the current configuration: ")

    print("Loading Data")
    data = load_json_file(filepath)

    print("Preparing dataset")

    def preprocess_split(split):
        split = read_examples(split, mode=='train')
        split = shuffle(split)
        split["proc_doc_tokens"] = split["doc_tokens"].apply(preprocess_tokens)
        split["proc_quest_tokens"] = split["quest_tokens"].apply(preprocess_tokens)
        split["bert_tokenized_doc_tokens"] = split["doc_tokens"].apply(bert_tokenization, tokenizer=tokenizer)
        split["bert_tokenized_quest_tokens"] = split["quest_tokens"].apply(bert_tokenization, tokenizer=tokenizer)
        if (model_choice == "3"):
            split["pos_tag"], split["exact_lemma"], split["tf"] = get_additional_features(split, BERT_MAX_LEN)
        return split

    tokenizer = load_bert_tokenizer()

    if mode == 'train':
        train = data[:VAL_SPLIT_INDEX]
        eval = data[VAL_SPLIT_INDEX:]
        train = preprocess_split(train)
        eval = preprocess_split(eval)
    elif mode == 'test':
        test = preprocess_split(data)

    if (model_choice == "3"):
        print("Preparing additional features")
        X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, X_train_pos_tags, X_train_exact_lemma, X_train_tf, y_train_start, y_train_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train)
        X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, X_eval_pos_tags, X_eval_exact_lemma, X_eval_tf, y_eval_start, y_eval_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval)
    else:
        X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, y_train_start, y_train_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train, with_features=False)
        X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, y_eval_start, y_eval_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval, with_features=False)

    eval_lookup_list = compute_lookups(eval)
    X_val_qas_id = eval["qas_id"].values.tolist()

    X_train_input_ids, X_train_token_type_ids, X_train_attention_mask = pad_inputs(X_train_input_ids, X_train_token_type_ids, X_train_attention_mask)
    X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask = pad_inputs(X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask)

    X_train = [X_train_input_ids, X_train_token_type_ids, X_train_attention_mask]
    X_val = [X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask]

    if (model_choice == "3"):
        X_train_pos_tags, X_train_exact_lemma, X_train_tf = pad_additonal_features(X_train_pos_tags, X_train_exact_lemma, X_train_tf)
        X_eval_pos_tags, X_eval_exact_lemma, X_eval_tf = pad_additonal_features(X_eval_pos_tags, X_eval_exact_lemma, X_eval_tf)
        X_train.extend([X_train_pos_tags, X_train_exact_lemma, X_train_tf])
        X_val.extend([X_eval_pos_tags, X_eval_exact_lemma, X_eval_tf])
        
    y_train = [y_train_start, y_train_end]
    y_val = [y_eval_start, y_eval_end]

    print("Fitting data to generators")
    TRAIN_LEN = X_train[0].shape[0]
    VAL_LEN = X_val[0].shape[0]

    if (model_choice == "3"):
        train_generator = features_data_generator(X_train, y_train, BATCH_SIZE)
        val_generator = features_data_generator(X_val, y_val, BATCH_SIZE)
    else:
        train_generator = baseline_data_generator(X_train, y_train, BATCH_SIZE)
        val_generator = baseline_data_generator(X_val, y_val, BATCH_SIZE)

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

    exact_match_callback = ExactMatch(X_val, y_val, eval_doc_tokens, X_eval_input_ids, eval_orig_answer_text, eval_lookup_list)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    if mode == "test":
        print("\nLoading model weights:\n\n")
        MODEL_PATH = osp.join(weights_path, "weights.h5")
        model.load_weights(MODEL_PATH)

    elif mode == "train":  
        print("\nTrain start:\n\n")
        model.fit(
            train_generator,
            validation_data = val_generator,
            steps_per_epoch = TRAIN_LEN / BATCH_SIZE,
            validation_steps = VAL_LEN / BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[exact_match_callback, es],
            workers=WORKERS)
        print("### SAVING MODEL ###")
        model.save_weights(os.path.join(outputdir, "weights.h5"))
        print("Weights saved to: weights.h5 inside the model directory")

    # Compute predictions using the evaluation set
    out = model.predict(X_val)
    start_idx = np.argmax(out[0], axis=-1).tolist()
    end_idx = np.argmax(out[1], axis=-1).tolist()
    result = {}
    for i in range(len(X_val_qas_id)):
        idx = X_val_qas_id[i]
        start = start_idx[i]
        end = end_idx[i]
        try:          
            pred = " ".join(eval_doc_tokens[i][eval_lookup_list[i][start]:(eval_lookup_list[i][end] + 1)])
        except KeyError:
            pred = tokenizer.decode(X_eval_input_ids[i][start:end+1])
        result[idx] = pred
    json_object = json.dumps(result)
    with open(osp.join(outputdir, "predictions.txt"), "w") as outfile:
        outfile.write(json_object)

    return model
