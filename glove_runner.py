from glove.utils import *
from glove.data_preparation import *
from glove.additional_features_preparation import *
from glove.constants import *
from glove.callback import *
from glove.glove_embedding import *
from glove.models import *
from glove.layers import *
from glove.generators import *

FILEPATH = "SQUAD/training_set.json"

data = load_json_file(FILEPATH)

train = data[:VAL_SPLIT_INDEX]
eval = data[VAL_SPLIT_INDEX:]

print("Preparing dataset")
train = read_examples(train)
eval = read_examples(eval)

train.sample(frac=1).reset_index(drop=True)
eval.sample(frac=1).reset_index(drop=True)

print("Tokenization and processing")
train["proc_doc_tokens"] = train['doc_tokens'].apply(preprocess_tokens)
train["proc_quest_tokens"] = train['quest_tokens'].apply(preprocess_tokens)

eval["proc_doc_tokens"] = eval['doc_tokens'].apply(preprocess_tokens)
eval["proc_quest_tokens"] = eval['quest_tokens'].apply(preprocess_tokens)

train = remove_too_long_samples(train)
eval = remove_too_long_samples(eval)

train = remove_not_valid_answer(train)
eval = remove_not_valid_answer(eval)

print("Preparing embedding")
embedding_model = load_embedding_model()
embedding_model = add_oov_words(train, embedding_model)
word2index, index2word = build_embedding_indices(embedding_model)

print("Processing tokens to be fed into model")
X_train_quest, X_train_doc = embed_and_pad_sequences(train, word2index, embedding_model)
X_val_quest, X_val_doc = embed_and_pad_sequences(eval, word2index, embedding_model)

y_train_start = train.start_position.to_numpy()
y_train_end = train.end_position.to_numpy()
y_val_start = eval.start_position.to_numpy()
y_val_end = eval.end_position.to_numpy()
val_doc_tokens = eval.doc_tokens.to_numpy()
val_answer_text = eval.orig_answer_text.to_numpy()

print("Building additional features")
X_train_doc_tags, pos_number = build_pos_features(train)
X_train_exact_lemma = build_exact_lemma_features(train)
X_train_tf = build_term_frequency_features(train)

X_eval_doc_tags, pos_number = build_pos_features(eval)
X_eval_exact_lemma = build_exact_lemma_features(eval)
X_eval_tf = build_term_frequency_features(eval)

X_train = [X_train_quest, X_train_doc, X_train_doc_tags, X_train_exact_lemma, X_train_tf]
y_train = [y_train_start, y_train_end]
X_val = [X_val_quest, X_val_doc, X_eval_doc_tags, X_eval_exact_lemma, X_eval_tf]
y_val = [y_val_start, y_val_end]

print("Freeing up memory, cleaning unuesd variables")
del train
del eval
gc.collect()

print("Fitting data to generators")
TRAIN_LEN = X_train[0].shape[0]
VAL_LEN = X_val[0].shape[0]
BS = 32

train_generator = features_data_generator(X_train, y_train, 32)
val_generator = features_data_generator(X_val, y_val, 32)

print("Creating model:\n")
model = attention_with_features(embedding_model, pos_number)
model.summary()

exact_match_callback = ExactMatch(X_val, y_val, val_doc_tokens, val_answer_text)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

print("\nTrain start:\n\n")
history = model.fit(
    train_generator,
    validation_data = val_generator,
    steps_per_epoch = TRAIN_LEN/BS,
    validation_steps = VAL_LEN/BS,
    epochs=20,
    verbose=1,
    callbacks=[exact_match_callback, es],
    workers = 8
)