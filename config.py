import os.path as osp
from pickle import TRUE

# ##########
# FILE PATHS
# ##########
URL_EMBEDDING_MODEL = "https://drive.google.com/uc?export=download&id=1catlig-Ubt4ztedW_nag_y4FQGBiOebJ"
DATA_PATH = osp.join("data", "training_set.json")
GLOVE_WEIGHTS_PATH = osp.join("data", "glove", "weights")
BERT_WEIGHTS_PATH = osp.join("data", "bert", "weights")
BIDAF_WEIGHTS_PATH = osp.join("data", "bidaf", "weights")

# #########
# CONSTANTS
# #########
PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~–—'
EMPTY_TOKEN = '<EMPTY>'
PAD_TOKEN = '<PAD>'
PAD_POS = 'PAD'
VAL_SPLIT_INDEX = 375

# ###################
# GLOVE CONFIGURATION
# ###################
EMBEDDING_MODEL_TYPE = 'glove'
EMBEDDING_DIMENSION = 300
CONCAT_EMBEDDING_DIMENSION = 900
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

# ###################
# BIDAF CONFIGURATION
# ###################

CONV_LAYERS = [[150, 10],
               [150, 7],
               [150, 5],
               [150, 3]]
FULLY_CONNECTED_LAYERS = [1024, 1024]
MAX_WORD_LEN = 15
NUM_HIGHWAY = 2