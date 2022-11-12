from tensorflow.keras.layers import Input, Embedding, Attention, Multiply, Bidirectional, Activation, Dropout, Dense, LSTM, Concatenate, TimeDistributed, Flatten
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from bidafLike.highway import Highway
from config import CONCAT_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION

def buildBidafModel(X_train_doc, X_train_quest, X_train_doc_char, X_train_quest_char, embedding_matrix, doc_char_model, quest_char_model):

    inputs_doc = Input(shape=(X_train_doc.shape[1]), name="X_train_doc")#(max_word_len,)
    inputs_quest = Input(shape=(X_train_quest.shape[1]), name="X_train_quest")#(max_word_len,)

    inputs_doc_char = Input(shape=(X_train_doc_char[0].shape), name="X_train_doc_char")#(max_word_len,)
    inputs_quest_char = Input(shape=(X_train_quest_char[0].shape), name="X_train_quest_char")#(max_word_len,)

    embedding = Embedding(input_dim = embedding_matrix.shape[0], 
                        output_dim = EMBEDDING_DIMENSION, 
                        weights = [embedding_matrix],
                        input_length = X_train_doc.shape[1],#max_word_len 
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

    for i in range(2):
        highway_layer = Highway()
        question_layer = TimeDistributed(highway_layer)
        question_embedding = question_layer(question_embedding)
        passage_layer = TimeDistributed(highway_layer)
        passage_embedding = passage_layer(passage_embedding)

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION // 2, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    passage_attention_scores = Attention()([passage_embedding, question_embedding])
    question_attention_scores = Attention()([question_embedding, passage_embedding])

    passage_embedding_att = Multiply()([passage_embedding, passage_attention_scores])
    question_embedding_att = Multiply()([question_embedding, question_attention_scores])

    passage_embedding = Concatenate()([passage_embedding, passage_embedding_att])
    question_embedding = Concatenate()([question_embedding, question_embedding_att])

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    hidden_layer = Bidirectional(LSTM(CONCAT_EMBEDDING_DIMENSION, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    passage = Concatenate(axis = 1)([passage_embedding, question_embedding])

    start_logits = Dense(1, name="start_logit", use_bias=False)(passage)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(passage)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    return Model(inputs=[inputs_quest ,inputs_doc, inputs_quest_char, inputs_doc_char], outputs=[start_probs, end_probs])
