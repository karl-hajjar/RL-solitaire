import numpy as np
from random import shuffle

class Buffer(object):
	"""Buffer is a class implementing a memory buffer of fixed capacity. The buffer will be used to store data from the games played
	and then sample batchs to update the network. Each data point in the buffer is represented as a dictionnary, where each key repre-
	sents an attribute of the data, and each value the corresponding value of the attribute. The buffer will thus contain a list of 
	dictionnaries."""
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []


	def add(self, el):
		self.buffer.append(el)
		if len(self.buffer) > self.capacity:
			self.buffer.pop(0)


	def add_list(self, l):
		self.buffer += l 
		if len(self.buffer) > self.capacity:
			self.buffer = self.buffer[-self.capacity:]


	def sample(self, n_samples):
		# shuffled = self.buffer.copy()
		# shuffle(shuffled)
		shuffled_indexes = list(range(len(self.buffer)))
		shuffle(shuffled_indexes)
		return [self.buffer[index] for index in shuffled_indexes[:n_samples]]

	# def sample_batch(self, batch_size):
	# 	'''
	# 	Returns a batch as a dictionnary where each attribute data is represented as a batch
	# 	'''
	# 	shuffled = self.buffer.copy()
	# 	shuffle(shuffled)
	# 	return shuffled[:n_samples]

