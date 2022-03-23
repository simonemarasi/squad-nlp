from common.utils import *
from common.functions import *
from glove.data_preparation import *
from common.additional_features_preparation import *
from common.constants import *
from glove.callback import *
from glove.glove_embedding import *
from glove.models import *
from glove.layers import *
from glove.generators import *
import os
from argparse import ArgumentParser
from datetime import datetime

print("\n\n")

parser = ArgumentParser()
# parser.add_argument("-mode", 
#                     dest="mode", 
#                     required=True,
#                     choices=['train', 'test'],
#                     help="Select the mode to run the model in.",
#                     metavar="MODE")    

parser.add_argument("-df", "--data-file", 
                    dest="datafile", 
                    required=True,
                    help="Name of the json file for training or testing",
                    metavar="DIR")

parser.add_argument("-od", "--output-directory", 
                    dest="outputdir", 
                    required=True,
                    help="Name of the directory where the output from training is saved (weights and history)",
                    metavar="DIR")

parser.add_argument("-wf", "--weights_file", 
                    dest="weights", 
                    required=False,
                    help=".h5 file where the model weights are saved. Loaded to continue training or testing", metavar="weightfile.h5")

parser.add_argument("-lr", "--learning_rate",
                    type=float,
                    dest="learning_rate", 
                    default=5e-5,
                    required=False,
                    help="Learning rate of the optimizer",
                    metavar="XX")

parser.add_argument("-bs", "--batch_size",
                    type=int,
                    dest="batch_size", 
                    required=True,
                    help="Batch size for the fit function",
                    metavar="XX")

parser.add_argument("-e", "--epochs",
                    type=int,
                    dest="epochs", 
                    required=True,
                    help="Number of training epochs",
                    metavar="XX")
                    
parser.add_argument("-w", "--workers", 
                    dest="workers",
                    type=int,
                    default=1, 
                    required=False,
                    help="Number of workers to use in the fit function", metavar="XX")

args = vars(parser.parse_args())

print("Loading Data")
FILEPATH = args['datafile']
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

train_generator = features_data_generator(X_train, y_train, args['batch_size'])
val_generator = features_data_generator(X_val, y_val, args['batch_size'])

print("Creating model:\n")
model = attention_with_features(args['learning_rate'], embedding_model, pos_number)
model.summary()

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")

exact_match_callback = ExactMatch(X_val, y_val, val_doc_tokens, val_answer_text)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

print("\nTrain start:\n\n")
history = model.fit(
    train_generator,
    validation_data = val_generator,
    steps_per_epoch = TRAIN_LEN / args['batch_size'],
    validation_steps = VAL_LEN / args['batch_size'],
    epochs=args['epochs'],
    verbose=1,
    callbacks=[exact_match_callback, es],
    workers = args['workers']
)

print("### SAVING MODEL ###")
model.save_weights(os.path.join(args['outputdir'],'weights-{}.h5'.format(now)))
print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
print("")
print("### SAVING HISTORY ###")
df_hist = pd.DataFrame.from_dict(history.history)
df_hist.to_csv(os.path.join(args['outputdir'],'history-{}.csv'.format(now)), mode='w', header=True)