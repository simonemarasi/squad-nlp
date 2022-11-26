from tensorflow.keras.layers import Input, Embedding, Attention, Multiply, Bidirectional, Dropout, Dense, LSTM, Concatenate, TimeDistributed
from tensorflow.keras import Model
from bidafLike.highway import Highway
from common.layers import *
from config import CONCAT_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION, MAX_CONTEXT_LEN, MAX_QUEST_LEN, MAX_WORD_LEN

def buildBidafModel(embedding_matrix, doc_char_model, quest_char_model):

    inputs_doc = Input(shape=(MAX_CONTEXT_LEN), name="X_doc")
    inputs_quest = Input(shape=(MAX_QUEST_LEN), name="X_quest")

    inputs_doc_char = Input(shape=((MAX_CONTEXT_LEN, MAX_WORD_LEN)), name="X_doc_char")
    inputs_quest_char = Input(shape=((MAX_QUEST_LEN, MAX_WORD_LEN)), name="X_quest_char")

    embedding = Embedding(input_dim = embedding_matrix.shape[0], 
                        output_dim = EMBEDDING_DIMENSION, 
                        weights = [embedding_matrix],
                        input_length = MAX_CONTEXT_LEN,
                        trainable = False,
                        name = "word_embedding")

    downsample = Dense(CONCAT_EMBEDDING_DIMENSION)

    passage_embedding = embedding(inputs_doc)
    passage_embedding_char = doc_char_model(inputs_doc_char)
    passage_embedding_char = downsample(passage_embedding_char)

    question_embedding = embedding(inputs_quest)
    question_embedding_char = quest_char_model(inputs_quest_char)
    question_embedding_char = downsample(question_embedding_char)

    passage_embedding = Concatenate()([passage_embedding, passage_embedding_char])
    question_embedding = Concatenate()([question_embedding, question_embedding_char])

    passage_embedding = Dropout(0.1)(passage_embedding)
    question_embedding = Dropout(0.1)(question_embedding)

    for _ in range(2):
        highway_layer = Highway()
        question_layer = TimeDistributed(highway_layer)
        question_embedding = question_layer(question_embedding)
        passage_layer = TimeDistributed(highway_layer)
        passage_embedding = passage_layer(passage_embedding)

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION // 2, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    quest_model, _ = WeightedSumAttention()(question_embedding)

    passage_attention_scores = Attention()([passage_embedding, question_embedding])

    passage_embedding_att = Multiply()([passage_embedding, passage_attention_scores])

    passage_embedding = Concatenate()([passage_embedding, passage_embedding_att])

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION, return_sequences=True))
    passage_embedding = hidden_layer(passage_embedding)

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION, return_sequences=True))
    passage_embedding = hidden_layer(passage_embedding)

    logits = BilinearSimilarity(CONCAT_EMBEDDING_DIMENSION)(quest_model, passage_embedding)

    start_probs, end_probs = Prediction()(logits)

    return Model(inputs=[inputs_quest ,inputs_doc, inputs_quest_char, inputs_doc_char], outputs=[start_probs, end_probs])
