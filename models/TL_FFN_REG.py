import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from tqdm import tqdm

from data_loader import DataLoader
from batch_loader import BatchLoader
from triplet_selector import TripletSelector
from model_data_visualizer import ModelDataVisualizer, visualize_loss

RANDOM_SEED = 73
torch.manual_seed(RANDOM_SEED)

def get_identifier(triplet_label: str, l2: float, dropout: float, lr: float, margin: float, batch_size: int, n_neighbours: int, min_dist: float):
    return f"TL_FFN_REG_{triplet_label}_{l2}L2_{dropout}D_{lr}LR_{margin}Margin_{batch_size}BatchSize_{n_neighbours}Neighbours_{min_dist}MinDist"


# ##################
# Model definition #
####################

class Model(nn.Module):
	def __init__(self, dropout: float):
		super(Model, self).__init__()
		self.initialize_model(dropout)

	def initialize_model(self, dropout: float):
		# 12 video features in
		self.l1 = nn.Linear(12,12)
		self.r1 = nn.ReLU()
		self.d1 = nn.Dropout(dropout)

		# 32 dimensions -> 32 dimensions
		self.l2 = nn.Linear(12,12)
		self.r2 = nn.ReLU()
		self.d2 = nn.Dropout(dropout)

		# 32 dimensions -> 16 output dimensions
		self.l3 = nn.Linear(12,12)

	def forward(self, x):
		# goes through all layers
		out = self.l1(x)
		out = self.r1(out)
		out = self.d1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.d2(out)
		out = self.l3(out)
		return out

def train(dataloader: DataLoader, device: str, options: dict) -> list:
	# Defaults
	lr = options.get("learning_rate", 0.0001)
	margin = options.get("margin", 1)
	l2 = options.get("l2", 1)
	dropout = options.get("dropout", 0.5)

	batch_size = options.get("batch_size", 1000)
	n_epoch = options.get("n_epoch", 25)
	start_epoch = options.get("start_epoch", 0)

	n_neighbours = options.get("n_neighbours", 5)
	min_dist = options.get("min_dist", 0.25)
	triplet_label = options.get("triplet_label")

	identifier = f"TL_FFN_REG_{triplet_label}_{l2}L2_{dropout}D_{lr}LR_{margin}Margin_{batch_size}BatchSize_{n_neighbours}Neighbours_{min_dist}MinDist"
	
	os.makedirs(f"model_checkpoints/{identifier}", exist_ok=True)

	model = Model(dropout=dropout).to(device)

	# Code
	losses = []	
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
	criterion = nn.TripletMarginLoss(margin=margin)

	X_train = dataloader.get_train_data()
	y_train = dataloader.get_train_labels()
	X_test = dataloader.get_test_data()
	y_test = dataloader.get_test_labels()

	np.random.seed(RANDOM_SEED)

	triplet_selector = TripletSelector("data/triplets.pkl", dataloader.label_cols)
	train_triplets = triplet_selector.standard(X_train, y_train, triplet_label)
	test_triplets = triplet_selector.standard(X_test, y_test, triplet_label)
	
	batch_loader = BatchLoader(X_train, train_triplets, batch_size)
	test_batch_loader = BatchLoader(X_test, test_triplets, batch_size)
	test_batch_loader.shuffle = False

	num_batches = batch_loader.num_of_batches()
	# batches x batch size x 3 x features (12)
	
	if start_epoch -1:
		try:
			checkpoint_cpu = torch.load(f"model_checkpoints/{identifier}/TL_FFN_last.pth", map_location=torch.device("cpu"), weights_only=False)
			start_epoch = checkpoint_cpu['epoch']
		except:
			start_epoch = 0
 
	if start_epoch > 0:
		print(f"starting at epoch {start_epoch}")
		checkpoint = torch.load(f"model_checkpoints/{identifier}/TL_FFN_epoch{start_epoch}.pth", map_location=torch.device(device), weights_only=False)
		checkpoint_cpu = torch.load(f"model_checkpoints/{identifier}/TL_FFN_epoch{start_epoch}.pth", map_location=torch.device("cpu"), weights_only=False)
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		losses = checkpoint_cpu['losses']

	for epoch in range(n_epoch):
		if epoch < start_epoch: continue
		
		model.train()
		criterion.train()
		batch_iter = iter(batch_loader)

		train_losses = []
		for batch in tqdm(batch_iter, total=num_batches, desc="Batch"):
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
			train_losses.append(loss.item())
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch+1}/{n_epoch}")

		model.eval()
		criterion.eval()
  
		# Add batchloader for test, and then average loss of each test batch
		test_losses = []
		test_batch_iter = iter(test_batch_loader)
		with torch.no_grad():
			for test_batch in test_batch_iter:
				test_anchor = model(torch.tensor(test_batch[:,0], dtype=torch.float).to(device))
				test_positive = model(torch.tensor(test_batch[:,1], dtype=torch.float).to(device))
				test_negative = model(torch.tensor(test_batch[:,2], dtype=torch.float).to(device))
   
				test_loss = criterion(test_anchor, test_positive, test_negative).item()
				test_losses.append(test_loss)

		avg_train_loss = np.mean(train_losses)
		avg_test_loss = np.mean(test_losses)

		losses.append([avg_train_loss, avg_test_loss])
		print(f"Train loss: {avg_train_loss}")
		print(f"Test loss: {avg_test_loss}")

		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch+1,
			'losses': losses
		}
		path = f"model_checkpoints/{identifier}/TL_FFN_epoch{epoch+1}.pth"
		lastpath = f"model_checkpoints/{identifier}/TL_FFN_last.pth"
		torch.save(checkpoint, path)
		torch.save(checkpoint, lastpath)
		print("MODEL SAVED")
		print("---------------------------------------")
	
		visualize_loss(
			np.array(losses),
			epoch+1,
			f"model_checkpoints/{identifier}/last_losses.svg",
			f"TL-FFN-REG - Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}"
		)

def visualize(dataloader: DataLoader, device: str, options: dict):
	lr = options.get("learning_rate", 0.0001)
	margin = options.get("margin", 1)
	l2 = options.get("l2", 1)
	dropout = options.get("dropout", 0.5)

	batch_size = options.get("batch_size", 1000)
	n_epoch = options.get("n_epoch", 25)

	n_neighbours = options.get("n_neighbours", 5)
	min_dist = options.get("min_dist", 0.25)
	triplet_label = options.get("triplet_label")

	identifier = f"TL_FFN_REG_{triplet_label}_{l2}L2_{dropout}D_{lr}LR_{margin}Margin_{batch_size}BatchSize_{n_neighbours}Neighbours_{min_dist}MinDist"
	
	os.makedirs(f"model_figures/{identifier}", exist_ok=True)

	X_train = dataloader.get_train_data()
	X_test = dataloader.get_test_data()
	y_train = dataloader.get_train_labels()
	y_test = dataloader.get_test_labels()

	# Train umap on raw input features and visualize train/test as before-train

	# visualizer = ModelDataVisualizer(
	# 		model,
	# 		device,
	# 		dataloader.label_cols,
	# 		triplet_label,
	# 		n_neighbours=n_neighbours,
	# 		min_dist=min_dist
	# )

	for epoch in tqdm(range(n_epoch), desc="Vis epoch"):
		model = Model(dropout=dropout).to(device)

		path = f"model_checkpoints/{identifier}/TL_FFN_epoch{epoch+1}.pth"
		try:
			checkpoint = torch.load(path, weights_only=False, map_location=torch.device(device))
		except:
			break
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()

		visualizer = ModelDataVisualizer(
			model,
			device,
			dataloader.label_cols,
			triplet_label,
			n_neighbours=n_neighbours,
			min_dist=min_dist
		)

		visualizer.fit(X_train)
		visualizer.UMAP_visualize(
			X_train,
			y_train,
			f"UMAP of train embedding - Epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_train.svg",
			subtitle_str=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)
		visualizer.UMAP_visualize(
			X_test,
			y_test,
			f"UMAP of test embedding - Epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_test.svg",
			subtitle_str=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)
  
		visualizer.UMAP_visualize_single(
			X_train,
			y_train,
			f"UMAP embedding of training representation - Epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_train_single.svg",
			subtitle_str=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)

		visualizer.UMAP_visualize_single(
			X_test,
			y_test,
			f"UMAP embedding of test representation - Epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_test_single.svg",
			subtitle_str=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)
  
		visualizer.compare_visualize(
			X_train,
			y_train,
			f"Comparison of training data UMAP visualizations: raw vs epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_comp_train.svg",
			fig_subtitle=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)
  
		visualizer.compare_visualize(
			X_test,
			y_test,
			f"Comparison of test data UMAP visualizations: raw vs epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_comp_test.svg",
			fig_subtitle=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"
		)
  
		visualizer.overlay(
			X_train,
			y_train,
			X_test,
			y_test,
			f"Overlay embedded data at epoch {epoch+1}",
			f"model_figures/{identifier}/TL_FFN_umap_epoch{epoch+1}_overlayed.svg",
			fig_subtitle=f"TL-FFN-REG params: Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}\nUMAP params: N-neighbours={n_neighbours} - Min distance={min_dist}"

		)

		visualize_loss(
			np.array(checkpoint["losses"]),
			epoch+1, f"model_figures/{identifier}/last_losses.svg",
			f"TL-FFN-REG - Label={triplet_label} - LR={lr} - Triplet margin={margin} - L2={l2} - Dropout={dropout}"
		)