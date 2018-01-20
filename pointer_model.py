from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, LSTMCell

class Pointer_Model(object):
	"""docstring for Question_Generator"""
	def __init__(self, config):
		self.task = config.task
	    self.debug = config.debug
	    self.config = config

	    self.input_dim = config.input_dim
	    self.hidden_dim = config.hidden_dim
	    self.num_layers = config.num_layers

	    self.max_enc_length = config.max_enc_length
	    self.max_dec_length = config.max_dec_length
	    self.num_glimpse = config.num_glimpse

	    self.init_min_val = config.init_min_val
	    self.init_max_val = config.init_max_val

	    self.initializer = tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

	    self.use_terminal_symbol = config.use_terminal_symbol

	    self.lr_start = config.lr_start
	    self.lr_decay_step = config.lr_decay_step
	    self.lr_decay_rate = config.lr_decay_rate
	    self.max_grad_norm = config.max_grad_norm

	    self.layer_dict = {}
	    
	    self.encoder_inputs = tf.placeholder()
	    self.decoder_targets = tf.placeholder()
	    self.encoder_seq_length = tf.placeholder()
	    self.decoder_seq_length = tf.placeholder()
	    self.mask = tf.placeholder()

	    if self.use_terminal_symbol:
	        self.decoder_seq_length +=1
		

	def _build_model(self, inputs):
		self.global_step = tf.Variable(0, trainable = False)

		with tf.variable_scope('embedding'):
			self.embedding = tf.get_variable(name = 'embedding' , shape = [self.vocab_size, self.embedding_dim],  
				initializer = self.initializer)

			self.embedding_lookup = tf.nn.embedding_lookup()

		with tf.variable_scope('encoder'):
			encoder_cell = LSTMCell(self.hidden_dim,
				initializer = self.initializer)

			if self.num_layers>1:
				cells = [encoder_cell] * self.num_layers
        			encoder_cell = MultiRNNCell(cells)			

			self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, sequence_length = self.encoder_seq_length)


		with tf.variable_scope('decoder'):
			self.decoder_cell = LSTMCell(self.hidden_dim, initializer = self.initializer)

			if self.num_layers > 1:
		        	cells = [self.decoder_cell] * self.num_layers
		        	self.decoder_cell = MultiRNNCell(cells)

		         #self.decoder_rnn = tf.contrib.seq2seq.Decoder()

		         self.decoder_pred_logits , _ = decoder_rnn.step(, )

		          self.dec_pred_prob = tf.nn.softmax(
		          	self.dec_pred_logits, 2, name="dec_pred_prob")

		         self.dec_pred = tf.argmax(
		         	self.dec_pred_logits, 2, name="dec_pred")

		with tf.variable_scope('decoder', reuse = True):
			self.decoder_pred_logits , _ = 

			self.dec_inference_prob = tf.nn.softmax(
		          self.dec_inference_logits, 2, name="dec_inference_logits")
		    	self.dec_inference = tf.argmax(
		          self.dec_inference_logits, 2, name="dec_inference")

		    

	def _build_optim(self):











		

