from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Concatenate, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import softmax
import tensorflow as tf
from transformers import TFBertModel
from common.constants import BERT_MAX_LEN, BERT_MODEL, LSTM_UNITS

def baseline_model(lr):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(BERT_MODEL)

    ## QA Model
    input_ids = Input(shape=(BERT_MAX_LEN, ), dtype=tf.int32)
    token_type_ids = Input(shape=(BERT_MAX_LEN, ), dtype=tf.int32)
    attention_mask = Input(shape=(BERT_MAX_LEN, ), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training = False)[0]

    embedding.trainable = False

    start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model

def baseline_with_rnn(lr):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(BERT_MODEL)

    ## QA Model
    input_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    token_type_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    attention_mask = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training = False)[0]

    embedding.trainable = False

    rnn = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(embedding)

    start_logits = Dense(1, name="start_logit", use_bias=False)(rnn)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(rnn)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(lr=lr, clipnorm = 1)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    
    return model

def features_with_rnn(lr):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(BERT_MODEL)

    ## QA Model
    input_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    token_type_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    attention_mask = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32)
    input_pos = Input(shape=(BERT_MAX_LEN, 46,))
    input_exact_lemmas = Input(shape=(BERT_MAX_LEN, 2,))
    input_term_frequency = Input(shape=(BERT_MAX_LEN, 1,))
    
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training = False)[0]

    embedding.trainable = False

    concatenate = Concatenate(axis = 2)([embedding, input_pos, input_exact_lemmas, input_term_frequency])

    rnn = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True))(concatenate)

    start_logits = Dense(1, name="start_logit", use_bias=False)(rnn)
    start_logits = Flatten()(start_logits)

    end_logits = Dense(1, name="end_logit", use_bias=False)(rnn)
    end_logits = Flatten()(end_logits)

    start_probs = Activation(softmax)(start_logits)
    end_probs = Activation(softmax)(end_logits)

    model = Model(
        inputs=[input_ids, token_type_ids, attention_mask, input_pos, input_exact_lemmas, input_term_frequency],
        outputs=[start_probs, end_probs],
    )
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = Adam(lr=lr, clipnorm=1)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model