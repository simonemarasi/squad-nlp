from common.constants import LR, LSTM_UNITS, MAX_CONTEXT_LEN, MAX_QUEST_LEN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Concatenate, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
from glove.layers import *

def baseline_model(learning_rate, embedding_model):

    embedding = embedding_model.get_keras_embedding(train_embeddings=False)

    input_quest = Input(shape=(MAX_QUEST_LEN,))
    quest_model = embedding(input_quest)
    quest_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(quest_model)

    input_context = Input(shape=(MAX_CONTEXT_LEN,))
    context_model = embedding(input_context)
    context_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(context_model)

    model = Concatenate(axis = 1)([context_model, quest_model])

    start_logits = Dense(1, name="start_logit", use_bias=False)(model)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(model)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(inputs=[input_quest, input_context],
                  outputs=[start_probs, end_probs])

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model

def attention_model(learning_rate, embedding_model):
    embedding = embedding_model.get_keras_embedding(train_embeddings=False)

    input_quest = Input(shape=(MAX_QUEST_LEN,))
    quest_model = embedding(input_quest)
    quest_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(quest_model)

    quest_model, _ = Attention()(quest_model)

    input_context = Input(shape=(MAX_CONTEXT_LEN,))
    context_model = embedding(input_context)
    context_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(context_model)

    logits = BilinearSimilarity(LSTM_UNITS)(quest_model, context_model)

    start_probs, end_probs = Prediction()(logits)

    model = Model(
            inputs=[input_quest, input_context],
            outputs=[start_probs, end_probs],
        )

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])
  
    return model

def baseline_with_features(learning_rate, embedding_model, pos_number):

    embedding = embedding_model.get_keras_embedding(train_embeddings=False)

    input_quest = Input(shape=(MAX_QUEST_LEN,))
    quest_model = embedding(input_quest)
    quest_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(quest_model)

    input_context = Input(shape=(MAX_CONTEXT_LEN,))
    input_pos = Input(shape=(MAX_CONTEXT_LEN, pos_number,))
    input_exact_lemma = Input(shape=(MAX_CONTEXT_LEN, 2, ))
    input_tf = Input(shape=(MAX_CONTEXT_LEN, 1, ))

    context_embedding = embedding(input_context)
    context_model = Concatenate(axis = 2)([context_embedding, input_pos, input_exact_lemma, input_tf])
    context_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(context_model)

    model = Concatenate(axis = 1)([context_model, quest_model])

    start_logits = Dense(1, name="start_logit", use_bias=False)(model)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(model)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(
            inputs=[input_quest, input_context, input_pos, input_exact_lemma, input_tf],
            outputs=[start_probs, end_probs])

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model

def attention_with_features(learning_rate, embedding_model, pos_number):
    embedding = embedding_model.get_keras_embedding(train_embeddings=False)

    input_quest = Input(shape=(MAX_QUEST_LEN,))
    quest_model = embedding(input_quest)
    quest_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(quest_model)

    quest_model, _ = Attention()(quest_model)

    input_context = Input(shape=(MAX_CONTEXT_LEN,))
    input_pos = Input(shape=(MAX_CONTEXT_LEN, pos_number,))
    input_exact_lemma = Input(shape=(MAX_CONTEXT_LEN, 2, ))
    input_tf = Input(shape=(MAX_CONTEXT_LEN, 1, ))

    context_embed = embedding(input_context)
    context_model = Concatenate(axis = 2)([context_embed, input_pos, input_exact_lemma, input_tf])
    context_model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(context_model)

    logits = BilinearSimilarity(LSTM_UNITS)(quest_model, context_model)

    start_probs, end_probs = Prediction()(logits)

    model = Model(
            inputs=[input_quest, input_context, input_pos, input_exact_lemma, input_tf],
            outputs=[start_probs, end_probs],
        )

    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model