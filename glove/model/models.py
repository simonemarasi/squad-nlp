from config import GLOVE_LSTM_UNITS, MAX_CONTEXT_LEN, MAX_QUEST_LEN, EMBEDDING_DIMENSION, MAX_WORD_LEN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Embedding, Add, Attention, Multiply, Dropout, TimeDistributed, Dense, Bidirectional, LSTM, Input, Concatenate, Flatten, Activation
from tensorflow.keras import Model
from glove.model.highway import Highway
from tensorflow import reduce_sum

def build_model(embedding_matrix, learning_rate, num_highway_layers=2, features=False, attention=False, char_embedding=False, doc_char_model=None, quest_char_model=None):
    inputs_doc = Input(shape=(MAX_CONTEXT_LEN), name="X_doc")
    inputs_quest = Input(shape=(MAX_QUEST_LEN), name="X_quest")

    inputs = [inputs_quest, inputs_doc]

    if features:
        input_exact_lemma = Input(shape=(MAX_CONTEXT_LEN, 2, ))
        input_tf = Input(shape=(MAX_CONTEXT_LEN, 1, ))
        feature_attention_score = reduce_sum(input_exact_lemma, axis=2, keepdims=True)
        feature_attention_score = Add()([feature_attention_score, input_tf])
        inputs.extend([input_exact_lemma, input_tf])

    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                          output_dim=EMBEDDING_DIMENSION,
                          weights=[embedding_matrix],
                          input_length=MAX_CONTEXT_LEN,
                          trainable=False,
                          name="word_embedding")

    passage_embedding = embedding(inputs_doc)
    question_embedding = embedding(inputs_quest)

    if char_embedding:
        inputs_doc_char = Input(shape=((MAX_CONTEXT_LEN, MAX_WORD_LEN)), name="X_doc_char")
        inputs_quest_char = Input(shape=((MAX_QUEST_LEN, MAX_WORD_LEN)), name="X_quest_char")
        passage_embedding_char = doc_char_model(inputs_doc_char)
        question_embedding_char = quest_char_model(inputs_quest_char)
        passage_embedding = Concatenate()([passage_embedding, passage_embedding_char])
        question_embedding = Concatenate()([question_embedding, question_embedding_char])    

    if features:
        passage_embedding_features_att = Multiply()([passage_embedding, feature_attention_score])
        passage_embedding = Add()([passage_embedding, passage_embedding_features_att])

    passage_embedding = Dropout(0.1)(passage_embedding)
    question_embedding = Dropout(0.1)(question_embedding)

    for _ in range(num_highway_layers):
        highway_layer = Highway()
        question_layer = TimeDistributed(highway_layer)
        question_embedding = question_layer(question_embedding)
        passage_layer = TimeDistributed(highway_layer)
        passage_embedding = passage_layer(passage_embedding)

    hidden_layer = Bidirectional(LSTM(GLOVE_LSTM_UNITS, return_sequences=True))

    passage_embedding = hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    lstm_units = GLOVE_LSTM_UNITS
    if attention:
        passage_embedding_att = Attention()([passage_embedding, question_embedding])
        passage_embedding = Concatenate()([passage_embedding, passage_embedding_att])
        question_embedding_att = Attention()([question_embedding, passage_embedding])
        question_embedding = Concatenate()([question_embedding, question_embedding_att])
        lstm_units = GLOVE_LSTM_UNITS * 2

    hidden_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    hidden_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))

    passage_embedding =  hidden_layer(passage_embedding)
    question_embedding = hidden_layer(question_embedding)

    passage = Concatenate(axis=1)([passage_embedding, question_embedding])

    start_logits = Dense(1, name="start_logit", use_bias=False)(passage)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(passage)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(inputs=inputs, outputs=[start_probs, end_probs])

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model
