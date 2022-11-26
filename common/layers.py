import tensorflow as tf
from config import MAX_CONTEXT_LEN

class WeightedSumAttention(tf.keras.Model):
    """
    This class implements a weighted sum attention.
    """
    def __init__(self, units = 1):   
        super(WeightedSumAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        
    def call(self, values):
        # values is the output of encoder. shape of values (batch,max_len,units)
        # score shape will be (batch,max_len,1)
        score = self.W(values)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class BilinearSimilarity(tf.keras.Model):
    """
    This function calculates bilinear term used for answer span prediction.
    Reference took from --> https://github.com/kellywzhang/reading-comprehension/blob/master/attention.py
    """    
    def __init__(self, hidden_size):
        super(BilinearSimilarity,self).__init__()
        self.units = hidden_size * 2
        self.WS = tf.keras.layers.Dense(self.units)
        self.WE = tf.keras.layers.Dense(self.units)
        
    def call(self, query, values):   
        # query corresponds to final question context vector (batch_size,hidden)
        # values are output of decoder i.e context (batch_size,decoder_timesteps,hidden)

        ################ start prediction ######################
        start = self.WS(query)
        # adding time_slice to question (batch_size,1,hidden)
        hidden_start_time_axis = tf.expand_dims(start, -1)
        
        # squeeze removes time slice we added before
        # final shape = (batch_size,decoder_timesteps)
        start_ = tf.squeeze(tf.matmul(values, hidden_start_time_axis),-1)
        start_ = tf.nn.softmax(start_, axis = 1)
            
        ################ end prediction ######################
        end = self.WE(query)
        hidden_end_time_axis = tf.expand_dims(end, -1)
        
        # squeeze remooves time slice we added before
        # final shape = (batch_size,decoder_timesteps)
        end_ = tf.squeeze(tf.matmul(values, hidden_end_time_axis),-1)
        end_ = tf.nn.softmax(end_, axis=1)
        
        prob = tf.concat((start_,end_),axis = 1)
        return prob

class Prediction(tf.keras.Model):
    """
    This class predicts probabilities of start and end token.
    Took reference from https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/
    """
    def __init__(self,token_span = 15):
        super(Prediction,self).__init__()
        self.token_span = token_span
     
    def call(self, prob):
        start_prob = prob[:, :MAX_CONTEXT_LEN]
        end_prob = prob[:, MAX_CONTEXT_LEN:]
        # do the outer product
        outer = tf.matmul(tf.expand_dims(start_prob, axis=2),tf.expand_dims(end_prob, axis=1))
        outer = tf.compat.v1.matrix_band_part(outer, 0, self.token_span)
        # start_position will have shape of (batch_size,)
        start_position = tf.reduce_max(outer, axis=2)
        end_position = tf.reduce_max(outer, axis=1)
        
        return start_position, end_position