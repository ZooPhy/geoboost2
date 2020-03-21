'''
Recurrent Neural network model object for the NER task
'''
import sys
import logging
import tensorflow as tf
from tensorflow.contrib import rnn

MODEL_NAMES = ["BIRNN", "BILSTM", "BILSTMP", "BIGRU", "BIUGRNN"]

class ModelHypPrms(object):
    '''Class for storing model hyperparameters'''
    def __init__(self, model_type, input_size, num_classes, hid_dims, lrn_rate,
                 num_layers, use_crf, max_len=15000):
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes
        self.hid_dims = hid_dims
        self.lrn_rate = lrn_rate
        self.num_layers = num_layers
        self.use_crf = use_crf
        self.max_len = max_len

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class BiRNNModel(object):
    '''Bi-directional RNN models based on the model_type parameter'''
    def __init__(self, hps):
        logging.info("Creating model with hyperparameters: %s", hps)
        # create placeholders
        self.input_x = tf.placeholder(tf.float32, [None, hps.max_len, hps.input_size])
        self.input_y = tf.placeholder(tf.float32, [None, hps.max_len, hps.num_classes])
        self.length = tf.placeholder(tf.int32, [None])
        self.dropout = tf.placeholder(tf.float32)

        # # permute the max_len and batch_size because batch_size can vary
        inputs = tf.transpose(self.input_x, [1, 0, 2])

        # Check if length of hid_dims values are equal to num of layers
        hid_dims = [int(x) for x in hps.hid_dims.split(',')]
        if hps.num_layers != len(hid_dims):
            logging.error("ERROR: Number of hid_dims values should equal the number of layers")
            sys.exit(1)

        with tf.name_scope("RNN"):
            # LSTM cell declarations and operations
            def cell(dims):
                '''Creates RNN/GRU/LSTM cell to use in multiple layers.'''
                # RNN
                if hps.model_type == "BIRNN":
                    cell = rnn.BasicRNNCell(dims)
                # LSTM
                elif hps.model_type == "BILSTM":
                    cell = rnn.LSTMCell(dims, state_is_tuple=True)
                # LSTM with peepholes
                elif hps.model_type == "BILSTMP":
                    cell = rnn.LSTMCell(dims, state_is_tuple=True,
                                        use_peepholes=True)
                # GRU
                elif hps.model_type == "BIGRU":
                    cell = rnn.GRUCell(dims)
                # Update Gate RNN
                elif hps.model_type == "BIUGRNN":
                    cell = rnn.UGRNNCell(dims)
                # Bi-directional Layer Norm LSTM
                elif hps.model_type == "BILNLSTM":
                    cell = rnn.LayerNormBasicLSTMCell(dims)
                return rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            fwd_cell = rnn.MultiRNNCell([cell(hid_dims[i]) for i in range(hps.num_layers)],
                                        state_is_tuple=True)
            bwd_cell = rnn.MultiRNNCell([cell(hid_dims[i]) for i in range(hps.num_layers)],
                                        state_is_tuple=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, inputs,
                                                         time_major=True,
                                                         sequence_length=self.length,
                                                         dtype=tf.float32,
                                                         scope="birnn")

        # reshape outputs for multiplication with final layer weights and bias
        # 1) concatenate fwd and bwd cells 2) permute to original input format 3) reshape for Wx+b
        outputs = tf.concat(outputs, 2)
        outputs = tf.nn.dropout(outputs, self.dropout)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        outputs = tf.reshape(outputs, [-1, 2 * hid_dims[hps.num_layers-1]])

        # weight and bias for the output layer
        with tf.name_scope("OTPT"):
            weight = tf.Variable(tf.truncated_normal([2 * hid_dims[hps.num_layers-1], hps.num_classes],
                                                     stddev=0.01))
            bias = tf.Variable(tf.constant(0., shape=[hps.num_classes]))

        # make prediction and reshape to original format
        prediction = tf.matmul(outputs, weight) + bias
        self.prediction = tf.reshape(prediction, [-1, hps.max_len, hps.num_classes])

        # Needed for decoding CRF transitions
        self.tr_prms = tf.constant(0, tf.float32)

        # Check if CRF is to be used
        if not hps.use_crf:
            # If not using CRF, use predictions from the output layer
            # Step 1: determine cost per token
            cost_full = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                labels=self.input_y)
            # Step 2: calculate average cost calculated based on sentence lengths only
            self.cost = tf.reduce_mean(tf.boolean_mask(cost_full, tf.sequence_mask(self.length)))
        else:
            # If CRF is to be used on the outputs
            with tf.name_scope("CRF"):
                # Step 1: create transition matrix
                tr_mat = tf.Variable(tf.constant(0., shape=[hps.num_classes, hps.num_classes]))
                # Step 2: get label indices
                lab_ind = tf.cast(tf.argmax(self.input_y, -1), tf.int32)
                # Step 3: get the transition encoding and calculate likelihoods
                log_lklhd, tr_prms = tf.contrib.crf.crf_log_likelihood(self.prediction,
                                                                       lab_ind, self.length,
                                                                       tr_mat)
                # Step 4: calculate average of negative log likelihood
                self.cost = tf.reduce_mean(-log_lklhd)
                # Save transition encodings for decoding the values
                self.tr_prms = tr_prms
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hps.lrn_rate).minimize(self.cost)

