from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Convolution1D, GlobalMaxPooling1D, Embedding, AlphaDropout, TimeDistributed

class CharCNNModel(Model):

  def __init__(self, input_shape, char_embedding_matrix, embedding_size, max_word_len, num_classes, conv_layers, 
                dropout_p, fully_connected_layers, train_embedding = False):
    super().__init__()
    self.inputs = Input(shape=input_shape, name='sent_input', dtype='int64')
    self.embedding = Embedding(input_dim = char_embedding_matrix.shape[0], 
                      output_dim = embedding_size, 
                      weights = [char_embedding_matrix],
                      input_length = max_word_len, 
                      trainable = train_embedding,
                      name = "char_embedding")
    self.num_classes = num_classes
    self.dense = Dense(self.num_classes, activation="softmax")
    self.conv_layers = conv_layers
    self.fully_connected_layers = fully_connected_layers
    self.dropout_p = dropout_p

  def call(self, inputs, training=False):
    x = self.inputs(inputs)
    x = self.embedding(x)
    convolution_output = []
    for num_filters, filter_width in self.conv_layers:
        conv = TimeDistributed(Convolution1D(filters=num_filters,
                                kernel_size=filter_width,
                                activation='tanh',
                                name='Conv1D_{}_{}'.format(num_filters, filter_width)))(x)
        pool = TimeDistributed(GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width)))(conv)
        convolution_output.append(pool)
    x = Concatenate()(convolution_output)
    encoder_output = x
    x = Flatten()(x)
    for fl in self.fully_connected_layers:
        x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(self.dropout_p)(x)

    predictions = Dense(self.num_classes, activation="softmax")(x)

    if training:
        return predictions
        #metrics=["accuracy"]
    else:
        return encoder_output

"""
def charCnnModel(input_shape, embedding_size,
                 conv_layers, fully_connected_layers,
                 dropout_p, num_classes, optimizer='adam', loss='categorical_crossentropy', include_top = True, train_embedding = False):

        # Input layer
        inputs = Input(shape=input_shape, name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(input_dim = char_embedding_matrix.shape[0], 
                      output_dim = embedding_size, 
                      weights = [char_embedding_matrix],
                      input_length = max_word_len, 
                      trainable = train_embedding,
                      name = "char_embedding")(inputs)
        # Convolution layers
        convolution_output = []
        for num_filters, filter_width in conv_layers:
            conv = TimeDistributed(Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width)))(x)
            pool = TimeDistributed(GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width)))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        encoder_output = x
        x = Flatten()(x)
        # Fully connected layers
        for fl in fully_connected_layers:
            x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(dropout_p)(x)
            
        predictions = Dense(num_classes, activation="softmax")(x)

        # Build and compile model
        if include_top:
          model = Model(inputs=inputs, outputs=predictions)
          metrics=["accuracy"]
          model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
           model = Model(inputs=inputs, outputs=encoder_output)
           model.compile(optimizer=optimizer, loss=None)

        print("CharCNNKim model built: ")
        return model
        """