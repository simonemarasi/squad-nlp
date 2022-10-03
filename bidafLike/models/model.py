from tensorflow.keras.layers import Input, Embedding, Attention, Multiply, Bidirectional, Activation, Dropout, Dense, LSTM, Concatenate, TimeDistributed, Flatten
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from bidafLike.highway import Highway
from bidafLike.models.charCnnModel import CharCNNModel
from config import CONV_LAYERS, FULLY_CONNECTED_LAYERS, MAX_WORD_LEN, NUM_HIGHWAY

class BidafLikeModel(Model):

    def __init__(self, X_train_doc, X_train_quest, X_train_doc_char, X_train_quest_char, embedding_matrix, embedding_dim, concatenated_embedding_dim):
        super().__init__()
        self.inputs_doc = Input(shape=(X_train_doc.shape[1]), name="X_train_doc")
        self.inputs_quest = Input(shape=(X_train_quest.shape[1]), name="X_train_quest")
        self.inputs_doc_char = Input(shape=(X_train_doc_char[0].shape), name="X_train_doc_char")
        self.inputs_quest_char = Input(shape=(X_train_quest_char[0].shape), name="X_train_quest_char")
        self.embedding = Embedding(input_dim = embedding_matrix.shape[0], 
                            output_dim = embedding_dim, 
                            weights = [embedding_matrix],
                            input_length = X_train_doc.shape[1],
                            trainable = False,
                            name = "word_embedding")
        self.downsample = Dense(concatenated_embedding_dim)
        self.doc_char_model = CharCNNModel(input_shape=X_train_doc_char[0].shape,
                                            char_embedding_matrix=X_train_doc_char,
                                            max_word_len=MAX_WORD_LEN,
                                            embedding_size=300,
                                            conv_layers=CONV_LAYERS,
                                            fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                            dropout_p=0.1,
                                            num_classes=0,
                                            train_embedding=True)
        self.quest_char_model = CharCNNModel(input_shape=X_train_quest_char[0].shape,
                                            char_embedding_matrix=X_train_quest_char,
                                            max_word_len=MAX_WORD_LEN,
                                            embedding_size=300,
                                            conv_layers=CONV_LAYERS,
                                            fully_connected_layers=FULLY_CONNECTED_LAYERS,
                                            dropout_p=0.1,
                                            num_classes=0,
                                            train_embedding=True)

    def call(self, inputs_doc, inputs_quest, inputs_doc_char, inputs_quest_char):
        #inputs_doc = self.inputs_doc(inputs_doc)
        #inputs_quest = self.inputs_quest(inputs_quest)
        #inputs_doc_char = self.inputs_doc_char(inputs_doc_char)
        #inputs_quest_char = self.inputs_quest_char(inputs_quest_char)

        passage_embedding = self.embedding(inputs_doc)
        passage_embedding_char = self.doc_char_model(inputs_doc_char)
        passage_embedding_char.compile(optimizer="adam", loss=None)
        passage_embedding_char.load_weights("CNN_Pretrain_Weights/CNN_150_FineTunedEmbedding")
        passage_embedding_char.trainable = False
        passage_embedding_char = self.downsample(passage_embedding_char)

        question_embedding = self.embedding(inputs_quest)
        question_embedding_char = self.quest_char_model(inputs_quest_char)
        question_embedding_char.compile(optimizer="adam", loss=None)
        question_embedding_char.load_weights("CNN_Pretrain_Weights/CNN_150_FineTunedEmbedding")
        question_embedding_char.trainable = False
        question_embedding_char = self.downsample(question_embedding_char)

        passage_embedding = Concatenate()([passage_embedding, passage_embedding_char])
        question_embedding = Concatenate()([question_embedding, question_embedding_char])
        concatenated_embedding_dim = passage_embedding.shape[-1]
        passage_embedding = Dropout(0.1)(passage_embedding)
        question_embedding = Dropout(0.1)(question_embedding)

        for _ in range(NUM_HIGHWAY):
            highway_layer = Highway()
            question_layer = TimeDistributed(highway_layer)
            question_embedding = question_layer(question_embedding)
            passage_layer = TimeDistributed(highway_layer)
            passage_embedding = passage_layer(passage_embedding)

        hidden_layer = Bidirectional(LSTM(concatenated_embedding_dim // 2, return_sequences=True))

        passage_embedding =  hidden_layer(passage_embedding)
        question_embedding = hidden_layer(question_embedding)

        passage_attention_scores = Attention()([passage_embedding, question_embedding])
        question_attention_scores = Attention()([question_embedding, passage_embedding])

        passage_embedding_att = Multiply()([passage_embedding, passage_attention_scores])
        question_embedding_att = Multiply()([question_embedding, question_attention_scores])

        passage_embedding = Concatenate()([passage_embedding, passage_embedding_att])
        question_embedding = Concatenate()([question_embedding, question_embedding_att])

        hidden_layer = Bidirectional(LSTM(concatenated_embedding_dim, return_sequences=True))

        passage_embedding =  hidden_layer(passage_embedding)
        question_embedding = hidden_layer(question_embedding)

        hidden_layer = Bidirectional(LSTM(concatenated_embedding_dim, return_sequences=True))

        passage_embedding =  hidden_layer(passage_embedding)
        question_embedding = hidden_layer(question_embedding)

        passage = Concatenate(axis = 1)([passage_embedding, question_embedding])

        start_logits = Dense(1, name="start_logit", use_bias=False)(passage)
        start_logits = Flatten()(start_logits)

        end_logits = Dense(1, name="end_logit", use_bias=False)(passage)
        end_logits = Flatten()(end_logits)

        start_probs = Activation(softmax)(start_logits)
        end_probs = Activation(softmax)(end_logits)
        
        return [start_probs, end_probs]