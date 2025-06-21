import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np

from data_loader import DataLoader
from batch_loader import BatchLoader
from triplet_selector import TripletSelector
from model_data_visualizer import ModelDataVisualizer, visualize_loss

RANDOM_SEED = 73
torch.manual_seed(RANDOM_SEED)

# ##################
# Model definition #
####################

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.initialize_model()

	def initialize_model(self):
		# 12 video features in
		self.l1 = nn.Linear(12,12)
		self.r1 = nn.ReLU()

		# 32 dimensions -> 32 dimensions
		self.l2 = nn.Linear(12,12)
		self.r2 = nn.ReLU()

		# 32 dimensions -> 16 output dimensions
		self.l3 = nn.Linear(12,12)

	def forward(self, x):
		# goes through all layers
		out = self.l1(x)
		out = self.r1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.l3(out)
		return out

def train(model: nn.Module, dataloader: DataLoader, triplet_label: str, device: str, options: dict) -> list:
	# Defaults
	lr = options.get("learning_rate", 0.0001)
	margin = options.get("margin", 1)
	
	batch_size = options.get("batch_size", 1000)
	n_epoch = options.get("n_epoch", 25)

	n_neighbours = options.get("n_neighbours", 5)
	min_dist = options.get("min_dist", 0.1)

	identifier = f"TL_FFN_{lr}LR_{margin}Margin_{batch_size}BatchSize_{n_neighbours}Neighbours_{min_dist}MinDist"
	
	os.makedirs(f"model_figures/{identifier}", exist_ok=True)
	os.makedirs(f"model_checkpoints/{identifier}", exist_ok=True)

	# Code
	losses = []	
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.TripletMarginLoss(margin=margin)

	X_train = dataloader.get_train_data()
	y_train = dataloader.get_train_labels()
	X_test = dataloader.get_test_data()
	y_test = dataloader.get_test_labels()

	np.random.seed(RANDOM_SEED)

	triplet_selector = TripletSelector("data/triplets.pkl", dataloader.label_cols)
	train_triplets = triplet_selector.standard(X_train, y_train, triplet_label)
	test_triplets = triplet_selector.standard(X_test, y_test, triplet_label)
	
	batch_loader = BatchLoader(train_triplets, batch_size)
	num_batches = batch_loader.num_of_batches()
	# batches x batch size x 3 x features (12)
	
	for epoch in range(n_epoch):
		model.train()
		criterion.train()
		batch_iter = iter(batch_loader)
		for batch_i, batch in enumerate(batch_iter):
			anchor_batch = batch[:,0]
			positive_batch = batch[:,1]
			negative_batch = batch[:,2]

			anchor = torch.tensor(anchor_batch, dtype=torch.float).to(device)
			positive = torch.tensor(positive_batch, dtype=torch.float).to(device)
			negative = torch.tensor(negative_batch, dtype=torch.float).to(device)

			anchor_out = model(anchor)
			positive_out = model(positive)
			negative_out = model(negative)

			loss = criterion(anchor_out, positive_out, negative_out)
			loss.backward()
			optimizer.step()

			# print(f"""Epoch: {epoch+1}/{n_epoch}\nBatch: {batch_i+1}/{num_batches}\nLoss: {loss.item()}\n-------------------------------------------""")

		print(f"Epoch {epoch+1}/{n_epoch}")

		model.eval()
		criterion.eval()
  
		train_anchors = model(torch.tensor(train_triplets[:,0], dtype=torch.float).to(device))
		train_positives = model(torch.tensor(train_triplets[:,1], dtype=torch.float).to(device))
		train_negatives = model(torch.tensor(train_triplets[:,2], dtype=torch.float).to(device))

		test_anchors = model(torch.tensor(test_triplets[:,0], dtype=torch.float).to(device))
		test_positives = model(torch.tensor(test_triplets[:,1], dtype=torch.float).to(device))
		test_negatives = model(torch.tensor(test_triplets[:,2], dtype=torch.float).to(device))

		train_losses = criterion(train_anchors, train_positives, train_negatives).item()
		test_losses = criterion(test_anchors, test_positives, test_negatives).item()
  
		losses.append([train_losses, test_losses])
		print(f"Train loss: {train_losses}")
		print(f"Test loss: {test_losses}")

		visualizer = ModelDataVisualizer(
			model,
			device,
			dataloader.label_cols,
			triplet_label,
			n_neighbours=n_neighbours,
			min_dist=min_dist
		)

		visualizer.fit(X_train)
		visualizer.visualize(X_train, y_train, f"UMAP - Epoch {epoch+1} - Train data", f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_train.svg")
		visualizer.visualize(X_test, y_test, f"UMAP - Epoch {epoch+1} - Test data", f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_test.svg")

		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch+1	
		}
		path = f"model_checkpoints/{identifier}/TL_FFN_epoch{epoch+1}.pth"
		torch.save(checkpoint, path)
		print("MODEL SAVED")
		print("---------------------------------------")
	
	losses = np.array(losses)
	visualize_loss(losses, n_epoch, f"model_figures/{identifier}/last_losses.svg")