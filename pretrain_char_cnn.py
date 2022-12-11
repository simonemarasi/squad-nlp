import pandas as pd
import numpy as np
from config import *
from glove.glove_embedding import prepare_embedding_model
from common.utils import list_to_dict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from glove.model.charCnnModel import buildCharCnnModel
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from glove.callbacks import LearningRateReducer

train = pd.read_csv(osp.join(CHAR_PRETRAIN_PATH, "train.csv"), header=None, names=["Label", "Title", "Content"])
test = pd.read_csv(osp.join(CHAR_PRETRAIN_PATH, "test.csv"), header=None, names=["Label", "Title", "Content"])
classes = ["World", "Sports", "Business", "Sci/Tech"]

def preprocess_tokens(token_list):
  items = []
  for item in token_list:
   item = item.lower().translate(str.maketrans('', '', PUNCTUATION))
   if item != "":
     items.append(item)
  return items

def whitespace_tokenize(text):
  return text.str.strip().str.split()

train[["Title", "Content"]] = train[["Title", "Content"]].apply(preprocess_tokens)
train[["Title", "Content"]] = train[["Title", "Content"]].apply(whitespace_tokenize)

test[["Title", "Content"]] = test[["Title", "Content"]].apply(preprocess_tokens)
test[["Title", "Content"]] = test[["Title", "Content"]].apply(whitespace_tokenize)

train["Label"] = train["Label"].apply(lambda x: x-1)
test["Label"] = test["Label"].apply(lambda x: x-1)

embedding_model = prepare_embedding_model(train, True)

alphabet = list(ALPHABET)
alphabet.extend([PAD_TOKEN, UNK_TOKEN])
index2char = list_to_dict(alphabet)
char2index = {value: key for (key, value) in index2char.items()}

char_embedding_matrix = np.zeros((len(index2char), EMBEDDING_DIMENSION))
for index in index2char:
  if index == char2index[PAD_TOKEN]:
    np.zeros(shape=(1, EMBEDDING_DIMENSION))
  elif index == char2index[UNK_TOKEN]:
    np.random.uniform(low=-4.0, high=4.0, size=(1, EMBEDDING_DIMENSION))
  else:
    char_embedding_matrix[index] = embedding_model[index2char[index]]

trainlist = [row["Title"] + row["Content"] for _, row in train.iterrows()]
testlist = [row["Title"] + row["Content"] for _, row in test.iterrows()]

X_train_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in trainlist]
X_test_char = [[[char2index[UNK_TOKEN] if c not in char2index else char2index[c] for c in w] for w in s] for s in testlist]

for i in range(len(X_train_char)):
  X_train_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_train_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])

for i in range(len(X_test_char)):
  X_test_char[i] = pad_sequences(maxlen=MAX_WORD_LEN, sequences=X_test_char[i], padding="post", truncating="post", value=char2index[PAD_TOKEN])

CHARPAD = np.array([char2index[PAD_TOKEN] for _ in range(MAX_WORD_LEN)])
MAX_CONTEXT_LEN = 75

X_train_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_train_char, padding="post", truncating="post", value=CHARPAD)
X_test_char = pad_sequences(maxlen=MAX_CONTEXT_LEN, sequences=X_test_char, padding="post", truncating="post", value=CHARPAD)
y_train = to_categorical(train["Label"], num_classes=len(classes))
y_test = to_categorical(test["Label"], num_classes=len(classes))

lrr = LearningRateReducer(0.7)
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = buildCharCnnModel(input_shape=(MAX_CONTEXT_LEN, MAX_WORD_LEN),
                    embedding_size=300,
                    char_embedding_matrix=char_embedding_matrix,
                    conv_layers=CONV_LAYERS,
                    fully_connected_layers=FULLY_CONNECTED_LAYERS,
                    num_classes=len(classes),
                    train_embedding=False)
K.set_value(model.optimizer.learning_rate, 5e-5)
model.fit(X_train_char, y_train, batch_size=32, validation_split=0.2, shuffle=True, epochs=20, workers=WORKERS, callbacks=[es, lrr])

fineTunedModel = buildCharCnnModel(input_shape=(MAX_CONTEXT_LEN, MAX_WORD_LEN),
                    embedding_size=300,
                    char_embedding_matrix=char_embedding_matrix,
                    conv_layers=CONV_LAYERS,
                    fully_connected_layers=FULLY_CONNECTED_LAYERS,
                    num_classes=len(classes),
                    train_embedding=True)
K.set_value(model.optimizer.learning_rate, 5e-6)
fineTunedModel.fit(X_train_char, y_train, batch_size=32, validation_split=0.2, shuffle=True, epochs=10, workers=WORKERS, callbacks=[es, lrr])

fineTunedModel.save_weights(CHAR_WEIGHTS_PATH)
