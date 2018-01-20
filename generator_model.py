from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
from tensorflow.contrib import seq2seq

class Question_Generator(object):
	def __init__(self):
		
		self.passage_lengths = tf.placeholder()
		self.start_index = tf.placeholder()
		self.stop_index = tf.placeholder()

		with tf.name_scope('word-repres'):
			self.passage_repres = tf.placeholder(tf.float32, [None,])




