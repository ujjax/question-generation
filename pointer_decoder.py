from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
from tensorflow.python.util import nest

dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder
simple_decoder_fn_train = seq2seq.simple_decoder_fn_train

def Decoder(cell, inputs, enc_outputs, enc_final_states,
			seq_lenth, hidden_dim, batch_size, is_train,
			num_glimpse, initializer = None, max_length = None):
	with tf.variable_scope('decoder-network') as scope:



		if is_train:
			decoder_fn = simple_decoder_fn_train(enc_final_states)

		else:
			maximum_length = tf.convert_to_tensor(max_length, tf.int32)

			def decoder_fn():
				cell_output = 





