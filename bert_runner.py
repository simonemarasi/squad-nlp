from ast import arg
from common.utils import *
from common.functions import *
from bert.data_preparation import *
from common.additional_features_preparation import *
from common.constants import *
from bert.callback import *
from bert.models import *
from bert.generators import *
import os
from argparse import ArgumentParser
from datetime import datetime
from sklearn.utils import shuffle

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

train = shuffle(train)
eval = shuffle(eval)

print("Tokenization and processing")
train["proc_doc_tokens"] = train['doc_tokens'].apply(preprocess_tokens)
train["proc_quest_tokens"] = train['quest_tokens'].apply(preprocess_tokens)

eval["proc_doc_tokens"] = eval['doc_tokens'].apply(preprocess_tokens)
eval["proc_quest_tokens"] = eval['quest_tokens'].apply(preprocess_tokens)

save_bert_tokenizer()
tokenizer = load_bert_tokenizer()

train["bert_tokenized_doc_tokens"] = train['doc_tokens'].apply(bert_tokenization, tokenizer=tokenizer)
train["bert_tokenized_quest_tokens"] = train['quest_tokens'].apply(bert_tokenization, tokenizer=tokenizer)

eval["bert_tokenized_doc_tokens"] = eval['doc_tokens'].apply(bert_tokenization, tokenizer=tokenizer)
eval["bert_tokenized_quest_tokens"] = eval['quest_tokens'].apply(bert_tokenization, tokenizer=tokenizer)

print("Building additional features (it may take a while...)")
X_train_doc_tags, pos_number = build_pos_features(train, BERT_MAX_LEN)
X_train_exact_lemma = build_exact_lemma_features(train, BERT_MAX_LEN)
X_train_tf = build_term_frequency_features(train, BERT_MAX_LEN)

X_eval_doc_tags, pos_number = build_pos_features(eval, BERT_MAX_LEN)
X_eval_exact_lemma = build_exact_lemma_features(eval, BERT_MAX_LEN)
X_eval_tf = build_term_frequency_features(eval, BERT_MAX_LEN)

# WITH FEATURES
X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, X_train_pos_tags, X_train_exact_lemmas, X_train_term_frequency, y_train_bert_start, y_train_bert_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train)
X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, X_eval_pos_tags, X_eval_exact_lemmas, X_eval_term_frequency, y_eval_bert_start, y_eval_bert_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval)

#WITHOUT FEATURES
X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, y_train_bert_start, y_train_bert_end, train_doc_tokens, train_orig_answer_text = unpack_dataframe(train, with_features=False)
X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, y_eval_bert_start, y_eval_bert_end, eval_doc_tokens, eval_orig_answer_text = unpack_dataframe(eval, with_features=False)

y_train_start = np.array(y_train_bert_start)
y_train_end = np.array(y_train_bert_end)
y_eval_start = np.array(y_eval_bert_start)
y_eval_end = np.array(y_eval_bert_end)

X_train_input_ids = pad_sequences(X_train_input_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
X_train_token_type_ids = pad_sequences(X_train_token_type_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
X_train_attention_mask = pad_sequences(X_train_attention_mask, padding='post', truncating='post', maxlen=BERT_MAX_LEN)

X_eval_input_ids = pad_sequences(X_eval_input_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
X_eval_token_type_ids = pad_sequences(X_eval_token_type_ids, padding='post', truncating='post', maxlen=BERT_MAX_LEN)
X_eval_attention_mask = pad_sequences(X_eval_attention_mask, padding='post', truncating='post', maxlen=BERT_MAX_LEN)

X_train = [X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, X_train_doc_tags, X_train_exact_lemma, X_train_tf]
y_train = [y_train_start, y_train_end]
X_val = [X_eval_input_ids, X_eval_token_type_ids, X_eval_attention_mask, X_eval_doc_tags, X_eval_exact_lemma, X_eval_tf]
y_val = [y_eval_start, y_eval_end]

train_lookup_list = compute_lookups(train)
eval_lookup_list = compute_lookups(eval)

print("Fitting data to generators")
TRAIN_LEN = X_train[0].shape[0]
VAL_LEN = X_val[0].shape[0]

train_generator = features_data_generator(X_train, y_train, args['batch_size'])
val_generator = features_data_generator(X_val, y_val, args['batch_size'])

print("Creating model:\n")
model = baseline_with_rnn(args['learning_rate'])
model.summary()

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")

exact_match_callback = ExactMatch(X_val, y_val, eval_doc_tokens, X_eval_input_ids, eval_orig_answer_text, eval_lookup_list)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

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