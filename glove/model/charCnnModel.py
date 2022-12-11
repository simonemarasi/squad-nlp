from tensorflow.keras.models import Model
from config import MAX_WORD_LEN
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Convolution1D, GlobalMaxPooling1D, Embedding, AlphaDropout, TimeDistributed


def buildCharCnnModel(input_shape, embedding_size, char_embedding_matrix, conv_layers, fully_connected_layers, dropout_p=0.1,
                      num_classes=0, optimizer='adam', loss='categorical_crossentropy', include_top=True, train_embedding=False):
    """
    Build and compile the Character Level CNN model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='sent_input', dtype='int64')
    # Embedding layers
    x = Embedding(input_dim=char_embedding_matrix.shape[0],
                  output_dim=embedding_size,
                  weights=[char_embedding_matrix],
                  input_length=MAX_WORD_LEN,
                  trainable=train_embedding,
                  name="char_embedding")(inputs)
    # Convolution layers
    convolution_output = []
    for num_filters, filter_width in conv_layers:
        conv = TimeDistributed(Convolution1D(filters=num_filters,
                                             kernel_size=filter_width,
                                             activation='tanh',
                                             name='Conv1D_{}_{}'.format(num_filters, filter_width)))(x)
        pool = TimeDistributed(GlobalMaxPooling1D(
            name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width)))(conv)
        convolution_output.append(pool)
    x = Concatenate()(convolution_output)

    if not include_top:
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss=None)
        return model

    x = Flatten()(x)
    # Fully connected layers
    for fl in fully_connected_layers:
        x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(dropout_p)(x)

    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
