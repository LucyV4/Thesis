import numpy as np
import math

NP_RANDOM_SEED = 73

class BatchLoader():
	batch_i = 0
    
	def __init__(self, X: np.ndarray, triplets: np.ndarray, batch_size: int = 128, epoch_data_size: int = None):
		self.random_gen = np.random.default_rng(NP_RANDOM_SEED)
		self.X = X
		self.triplets = triplets
		self.batch_size = batch_size
		self.epoch_data_size = epoch_data_size if epoch_data_size else len(triplets)
		self.shuffle = True

	def num_of_batches(self):
		return math.ceil(self.epoch_data_size/self.batch_size)

	def __iter__(self):
		if self.shuffle: self.random_gen.shuffle(self.triplets)
		self.batch_i = 0
		return self

	def __next__(self):
		start_i = self.batch_i * self.batch_size
		end_i = (self.batch_i+1) * self.batch_size
        
		if start_i >= self.epoch_data_size: raise StopIteration
		else: 
			self.batch_i += 1
			
			epoch_triplets = self.triplets[start_i:end_i]
			epoch_data = [(self.X[anchor], self.X[pos], self.X[neg]) for (anchor, pos, neg) in epoch_triplets]
			return np.array(epoch_data)