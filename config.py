import os.path as osp

# ##########
# FILE PATHS
# ##########
EMBEDDING_PATH = osp.join("data", "models", "glove", "embedding_model.pkl")
DATA_PATH = osp.join("data", "training_set.json")
GLOVE_WEIGHTS_PATH = osp.join("data", "models", "glove", "weights")
BERT_WEIGHTS_PATH = osp.join("data", "models", "bert", "weights")

# #########
# CONSTANTS
# #########
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~–—'
EMPTY_TOKEN = '<EMPTY>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_POS = 'PAD'
VAL_SPLIT_INDEX = 350

# ###################
# GLOVE CONFIGURATION
# ###################
EMBEDDING_DIMENSION = 300
MAX_CONTEXT_LEN = 400
MAX_QUEST_LEN = 40

# ##################
# BERT CONFIGURATION
# ##################
BERT_SAVE_DIR = 'bert_base_uncased'
BERT_MODEL = 'bert-base-uncased'
BERT_MAX_LEN = 429
TRANSL_DICT = {"'": "''", "-": '--', ':': ':', '(': '(', ')': ')', '.': '.', ',': ',', '`': '``', '$': '$'}

# ###################
# MODEL CONFIGURATION
# ###################
WORKERS = 4
LEARNING_RATE = 5e-5
EPOCHS = 25
BATCH_SIZE = 64
LSTM_UNITS = 250
GLOVE_LSTM_UNITS = 300

# ###################
# BIDAF CONFIGURATION
# ###################

CONV_LAYERS = [[100, 10],
               [100, 7],
               [100, 5],
               [100, 3]]
FULLY_CONNECTED_LAYERS = [1024, 1024]
CONCAT_EMBEDDING_DIMENSION = 600
MAX_WORD_LEN = 15
NUM_HIGHWAY = 2
CHAR_WEIGHTS_PATH = osp.join("data", "models", "glove", "weights", "char", "CNN_100_FineTunedEmbedding")
CHAR_PRETRAIN_PATH = osp.join("data", "ag_news")
LR_REDUCER_RATE = 0.8
