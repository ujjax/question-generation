from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib import seq2seq

class Question_Generator(object):
	def __init__(self):
		
		self.input_dim = input_dim
		self.start_index = tf.placeholder()
		self.stop_index = tf.placeholder()
		self.question = tf.placeholder()

		with tf.name_scope('word-repres'):
			self.passage_repres = tf.placeholder(tf.float32, [None,None,None])
			
			if with_char and char_vocab is not None:
				self.passage_char_lengths = tf.placeholder(tf.float32, [None,None])
				
				self.passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]
				input_shape = tf.shape(self.answer_chars)
				batch_size = input_shape[0]
				a_char_len = input_shape[2]
				input_shape = tf.shape(self.passage_chars)
				passage_len = input_shape[1]
				p_char_len = input_shape[2]
				char_dim = char_vocab.word_dim
				self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), 
					dtype=tf.float32)
				passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
				passage_char_repres = tf.reshape(passage_char_repres, shape=[-1, p_char_len, char_dim])
				passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
				with tf.variable_scope('char_lstm'):
					# lstm cell
					char_lstm_cell = LSTMCell(char_lstm_dim)
					# dropout
					if is_training: char_lstm_cell = DropoutWrapper(char_lstm_cell, 
						output_keep_prob=(1 - dropout_rate))
					char_lstm_cell = MultiRNNCell([char_lstm_cell])
					
					tf.get_variable_scope().reuse_variables()
					# passage representation
					passage_char_outputs = tf.nn.dynamic_rnn(char_lstm_cell, passage_char_repres, 
							sequence_length=passage_char_lengths,dtype=tf.float32)[0] # [batch_size*answer_len, q_char_len, char_lstm_dim]
					passage_char_outputs = passage_char_outputs[:,-1,:]
					passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, char_lstm_dim])
					
				passage_repres.append(passage_char_outputs)
				self.input_dim += char_lstm_dim

			self.passage_repres = tf.concat(2, self.passage_repres) # [batch_size, passage_len, dim]

		with tf,name_scope('encoder-1'):
			encoder_cell_f = LSTMCell(hidden_dim)
			encoder_cell_b = LSTMCell(hidden_dim)

			encoder_ouputs , _ = tf.nn.bidirectional_dynamic_rnn(encoder_cell_f, encoder_cell_b, self.passage_repres)

			h_d = tf.concat(encoder_ouputs, axis =2)

		with tf.name_scope('answer-encoding'):
			unstacked_h_d = tf.unstack(h_d)
			h_a_ = []

			for i in range(len(unstacked_h_d)):
				temp = unstacked_h_d[i]
				h_a_.append(temp[start_index[i],stop_index[i]])

			answer_encoder_f = LSTMCell(hidden_dim)
			answer_encoder_b = LSTMCell(hidden_dim)

			h_a , _ = tf.nn.bidirectional_dynamic_rnn(answer_encoder_f,answer_encoder_b,inputs = tf.stack(h_a_))

			h_a_argmax = tf.argmax(h_a,2)

		with tf.name_scope('cascading-cell'):
			cascading_cell_1 = LSTMCell(hidden_dim)
			cascading_cell_2 = LSTMCell(hidden_dim)

			def cascading_cells_condition(t,,state, tensor_):
				

				with tf.variable_scope('weights-cascading'):
					W_1 = tf.get_variable()
					W_2 = tf.get_variable()
					b_1 = tf.get_variable()
					b_2 = tf.get_variable()

				

				temp_input = tf.concat(h_d_i,h_a_argmax,cascading_cell_1_output)
				v_t = tf.matmul((tf.matmul(temp_input,W_1)+b_1),W_2) + b_2

				alpha_t = 


				t +=1
				return t,state,tensor_

			with tf.variable_scope('cascading-lstm'):
				tensor_ = tf.TensorArray(dtype = tf.float32, size = hidden_dim)

				condition = lambda p,q,r,s: tf.less(p, )
				body = lambda p,q,r,s: cascading_cells_condition(p,q,r,s)
				t = tf.constant(0)

				cascading_loop = tf.while_loop(cond =condition , body = body,
													loop_vars = (t, ))		

