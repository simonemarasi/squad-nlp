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
from bidafLike.models import charCnnModel
from bidafLike.models import model
from common import learningRateReducer

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3", "4"]:
            print("Sorry, your choice must be between the four allowed")
            continue
        else:
            break
    return value

def bidaf_runner(filepath, outputdir=BIDAF_WEIGHTS_PATH):

    print("###########################")
    print("#### BIDAF-LIKE RUNNER ####")
    print("###########################")
    
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

    #######################################

    alphabet_string = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet = []
    for char in alphabet_string:
        alphabet.append(char)

    index2char_list = alphabet
    index2char_list.append("<PAD>")
    index2char_list.append("<UNK>")

    index2char = list_to_dict(index2char_list)
    char2index = {value : key for (key, value) in index2char.items()}

    # create char_embedding_matrix 

    char_embedding_matrix = np.zeros((len(index2char), EMBEDDING_DIMENSION))
    for index in index2char:
        if index == char2index["<PAD>"]:
            np.zeros(shape=(1, EMBEDDING_DIMENSION))
        elif index == char2index["<UNK>"]:
            np.random.uniform(low=-4.0, high=4.0, size=(1, EMBEDDING_DIMENSION))
        else:
            char_embedding_matrix[index] = embedding_model[index2char[index]]

    X_train_quest_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c] for c in w] for w in s] for s in train.proc_quest_tokens]
    X_train_doc_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c] for c in w] for w in s] for s in train.proc_doc_tokens]

    X_val_quest_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c] for c in w] for w in s] for s in eval.proc_quest_tokens]
    X_val_doc_char = [[[char2index['<UNK>'] if c not in char2index else char2index[c] for c in w] for w in s] for s in eval.proc_doc_tokens]

    for i in range(len(X_train_quest_char)):
        X_train_quest_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_train_quest_char[i], padding="post", truncating="post", value=char2index['<PAD>'])

    for i in range(len(X_train_doc_char)):
        X_train_doc_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_train_doc_char[i], padding="post", truncating="post", value=char2index['<PAD>'])

    for i in range(len(X_val_quest_char)):
        X_val_quest_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_val_quest_char[i], padding="post", truncating="post", value=char2index['<PAD>'])

    for i in range(len(X_val_doc_char)):
        X_val_doc_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_val_doc_char[i], padding="post", truncating="post", value=char2index['<PAD>'])

    charpad = np.array([char2index['<PAD>'] for i in range(MAX_WORD_LEN)])

    X_train_quest_char = pad_sequences(maxlen=MAX_QUEST_LEN, sequences=X_train_quest_char, padding="post", truncating="post", value=charpad)
    X_train_doc_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_train_doc_char, padding="post", truncating="post", value=charpad)
    X_val_quest_char = pad_sequences(maxlen=MAX_QUEST_LEN, sequences=X_val_quest_char, padding="post", truncating="post", value=charpad)
    X_val_doc_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_val_doc_char, padding="post", truncating="post", value=charpad)

    embedding_matrix = np.zeros((len(word2index), EMBEDDING_DIMENSION))
    for word in word2index:
        embedding_matrix[word2index[word]] = embedding_model[word]

    #########################################

    X_train = [X_train_quest, X_train_doc, X_train_quest_char, X_train_doc_char]
    y_train = [y_train_start, y_train_end]
    X_val = [X_val_quest, X_val_doc, X_val_quest_char, X_val_doc_char]
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

    doc_char_model = charCnnModel.build_charCnn_model(input_shape=X_train_doc_char[0].shape,
                    embedding_size=300,
                    conv_layers=CONV_LAYERS,
                    fully_connected_layers=FULLY_CONNECTED_LAYERS,
                    dropout_p=0.1,
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    num_classes = 0,
                    char_embedding_matrix=char_embedding_matrix,
                    include_top = False,
                    train_embedding = True)

    doc_char_model.load_weights(CHAR_WEIGHTS_PATH)
    doc_char_model.trainable = False

    quest_char_model = charCnnModel.build_charCnn_model(input_shape=X_train_quest_char[0].shape,
                    embedding_size=300,
                    conv_layers=CONV_LAYERS,
                    fully_connected_layers=FULLY_CONNECTED_LAYERS,
                    dropout_p=0.1,
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    num_classes = 0,
                    char_embedding_matrix=char_embedding_matrix,
                    include_top = False,
                    train_embedding = True)

    quest_char_model.load_weights(CHAR_WEIGHTS_PATH)
    quest_char_model.trainable = False

    bidafModel = model.buildBidafModel(X_train_doc, X_train_quest, X_train_doc_char, X_train_quest_char, embedding_matrix, doc_char_model, quest_char_model)

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=5e-4)
    bidafModel.compile(optimizer=optimizer, loss=[loss, loss])

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    exact_match_callback = ExactMatch(X_val, y_val, val_doc_tokens, val_answer_text)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    lr = learningRateReducer.LearningRateReducer(0.8)

    print("\nTrain start:\n\n")
    history = bidafModel.fit(
        train_generator,
        validation_data = val_generator,
        steps_per_epoch = TRAIN_LEN / BATCH_SIZE,
        validation_steps = VAL_LEN / BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[exact_match_callback, es, lr],
        workers = WORKERS
    )

    print("### SAVING MODEL ###")
    bidafModel.save_weights(os.path.join(outputdir,'weights-{}.h5'.format(now)))
    print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
    print("")
    print("### SAVING HISTORY ###")
    df_hist = pd.DataFrame.from_dict(history.history)
    df_hist.to_csv(os.path.join(outputdir,'history-{}.csv'.format(now)), mode='w', header=True)
