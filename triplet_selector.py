import pickle
import numpy as np
import hashlib

class TripletSelector:
	def __init__(self, filename: str, label_cols: list):
		self.filename = filename
		self.label_cols: list = label_cols
		try:
			with open(self.filename, "rb") as file:
				self.saved:dict = pickle.load(file)
		except:
				self.saved: dict = {}
    
	def calc_triplets(self, hash: int, y: np.ndarray, triplet_label_i: int):
		positives = {}
		negatives = {}

		labels = y[:, triplet_label_i]
		for label_i, label in enumerate(labels):
			pos = np.where(labels == label)[0]
			positives[label_i] = pos[(pos != label_i)]
			negatives[label_i] = np.where(labels != label)[0]
		self.saved[hash] = (positives, negatives)
		with open(self.filename, "wb") as file:
			pickle.dump(self.saved, file)
    
	def standard(self, X: np.ndarray, y: np.ndarray, triplet_label: str):
		print("Creating triplets")
		triplet_label_index = self.label_cols.index(triplet_label)
		label_hash = hashlib.sha256(str(y[:, triplet_label_index]).encode()).hexdigest()
		indices = range(len(y))

		if not self.saved or not self.saved.get(label_hash, None):
			self.calc_triplets(label_hash, y, triplet_label_index)
		
		(positives, negatives) = self.saved[label_hash]

		triplets = []
		for anchor_index in indices:
			positive_indices = positives[anchor_index]
			negative_indices = negatives[anchor_index]
			for positive_index in positive_indices:
				for negative_index in negative_indices:
					triplets.append((anchor_index, positive_index, negative_index))

		print(f"{len(triplets)} triplets created")
		print("---------------------------------------")
		return np.array(triplets)