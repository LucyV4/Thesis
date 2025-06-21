import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader as nnDataLoader
from torch.utils.data import Dataset as nnDataset

import os
import numpy as np
from tqdm import tqdm

from data_loader import DataLoader
from model_data_visualizer import visualize_loss

def get_identifier(classification_label: str, l2: float, dropout: float, lr: float, batch_size: int):
    return f"DIRECT_{classification_label}_{l2}L2_{dropout}D_{lr}LR_{batch_size}BatchSize"

class Dataset(nnDataset):
	def __init__(self, X, y, device):
		self.X = torch.tensor(X, dtype=torch.float).to(device)
		self.y = torch.tensor(y, dtype=torch.long).to(device)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

class DirectClasisfier(nn.Module):
	def __init__(self, dropout: float, output_features: int):
		super(DirectClasisfier, self).__init__()
		self.initialize_model(dropout, output_features)

	def initialize_model(self, dropout: float, output_features: int):
		# 12 video features in -> 32 dimensions
		self.l1 = nn.Linear(12,32)
		self.r1 = nn.ReLU()
		self.d1 = nn.Dropout(dropout)

		# 32 dimensions -> 32 dimensions
		self.l2 = nn.Linear(32,32)
		self.r2 = nn.ReLU()
		self.d2 = nn.Dropout(dropout)

		# 32 dimensions -> 32 dimensions
		self.l3 = nn.Linear(32,32)
		self.r3 = nn.ReLU()
		self.d3 = nn.Dropout(dropout)

		# 32 dimensions -> 16 dimensions
		self.l4 = nn.Linear(32,16)
		self.r4 = nn.ReLU()
		self.d4 = nn.Dropout(dropout)

		# 16 dimensions to 5 dimension (UPDRS score)
		self.l5 = nn.Linear(16, output_features)

	def forward(self, x):
		# goes through all layers
		out = self.l1(x)
		out = self.r1(out)
		out = self.d1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.d2(out)
		out = self.l3(out)
		out = self.r3(out)
		out = self.d3(out)
		out = self.l4(out)
		out = self.r4(out)
		out = self.d4(out)
		out = self.l5(out)
		return out

def train(device: str, dataloader: DataLoader, options: dict):
	classification_label = options.get("classification_label")
	lr = options.get("learning_rate", 0.0001)
	l2 = options.get("l2", 0.01)
	batch_size = options.get("batch_size", 1000)
	dropout = options.get("dropout", 0.5)

	n_epoch = options.get("n_epoch", 25)

	weights = options.get("weights", None)
 
	label_index = dataloader.label_cols.index(classification_label)
    
	X_train = dataloader.get_train_data()
	y_train = np.array(dataloader.get_train_labels()[:,label_index], dtype=np.uint8)
	X_test = dataloader.get_test_data()
	y_test = np.array(dataloader.get_test_labels()[:, label_index], dtype=np.uint8)

	model = DirectClasisfier(dropout=dropout, output_features=len(np.unique(y_train))).to(device)
        
	criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
 
	dataset = Dataset(X_train, y_train, device)
	batchloader = nnDataLoader(dataset, batch_size=batch_size, shuffle=True)

	losses = []

	identifier = get_identifier(classification_label, l2, dropout, lr, batch_size)
	os.makedirs(f"model_checkpoints/{identifier}/", exist_ok=True)

	for epoch in range(n_epoch):
		model.train()
		for inputs, labels in tqdm(batchloader, desc="Batch"):
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch+1}/{n_epoch}")

		with torch.no_grad():
			train_data = torch.tensor(X_train, dtype=torch.float).to(device)
			train_labels = torch.tensor(y_train, dtype=torch.long).to(device)
			test_data = torch.tensor(X_test, dtype=torch.float).to(device)
			test_labels = torch.tensor(y_test, dtype=torch.long).to(device)
			
			train_outputs = model(train_data)
			test_outputs = model(test_data)
   
			train_loss = criterion(train_outputs, train_labels).item()
			test_loss = criterion(test_outputs, test_labels).item()

			print(f"Train loss: {train_loss}")
			print(f"Test loss: {test_loss}")

			losses.append([train_loss, test_loss])

		visualize_loss(
			np.array(losses),
			epoch+1,
			f"model_checkpoints/{identifier}/last_losses.svg",
			f"Embedding model downstream task loss"
		)

		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optim_state_dict': optimizer.state_dict(),
			'losses': losses
		}

		path = f"model_checkpoints/{identifier}/model_direct_dict_epoch{epoch+1}.pth"
		torch.save(checkpoint, path)
  
		print("Model saved")
		print("---------------------------------------")