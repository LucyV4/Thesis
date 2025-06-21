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

class Dataset(nnDataset):
	def __init__(self, X, y, device):
		self.X = torch.tensor(X, dtype=torch.float).to(device)
		self.y = torch.tensor(y, dtype=torch.long).to(device)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

class DownstreamClasisfier(nn.Module):
	def __init__(self, input_features: int, output_features: int):
		super(DownstreamClasisfier, self).__init__()
		self.initialize_model(input_features, output_features)
	
	def initialize_model(self, input_features: int, output_features: int):
		# Input features to output features
		self.l1 = nn.Linear(input_features, output_features)

	def forward(self, x):
		# goes through all layers
		out = self.l1(x)
		return out

def train(embedding_model: nn.Module, device: str, dataloader: DataLoader, embedding_model_identifier: str, options: dict):
	classification_label = options.get("classification_label")
	lr = options.get("learning_rate", 0.0001)
	l2 = options.get("l2", 1)
	batch_size = options.get("batch_size", 1000)
	weights = options.get("weights", None)

	n_epoch = options.get("n_epoch", 25)
 
	label_index = dataloader.label_cols.index(classification_label)
    
	embedding_model = embedding_model.to(device)
	embedding_model.eval()

	X_train = dataloader.get_train_data()
	y_train = np.array(dataloader.get_train_labels()[:,label_index], dtype=np.uint8)
	X_test = dataloader.get_test_data()
	y_test = np.array(dataloader.get_test_labels()[:, label_index], dtype=np.uint8)

	X_train = torch.tensor(X_train, dtype=torch.float).to(device)
	X_test = torch.tensor(X_test, dtype=torch.float).to(device)

	X_train: np.ndarray = embedding_model(X_train).cpu().detach().numpy()
	X_test: np.ndarray = embedding_model(X_test).cpu().detach().numpy()

	downstream_model = DownstreamClasisfier(input_features=X_train.shape[1], output_features=len(np.unique(y_train))).to(device)
    
	criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
	optimizer = optim.Adam(downstream_model.parameters(), lr=lr, weight_decay=l2)

	dataset = Dataset(X_train, y_train, device)
	batchloader = nnDataLoader(dataset, batch_size=batch_size, shuffle=True)

	losses = []
 
	os.makedirs(f"model_checkpoints/{embedding_model_identifier}/downstream", exist_ok=True)

	for epoch in range(n_epoch):
		downstream_model.train()
		for inputs, labels in tqdm(batchloader, desc="Batch"):
			outputs = downstream_model(inputs)
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
			
			train_outputs = downstream_model(train_data)
			test_outputs = downstream_model(test_data)
   
			train_loss = criterion(train_outputs, train_labels).item()
			test_loss = criterion(test_outputs, test_labels).item()

			print(f"Train loss: {train_loss}")
			print(f"Test loss: {test_loss}")

			losses.append([train_loss, test_loss])
		print("---------------------------------------")

		visualize_loss(
			np.array(losses),
			epoch+1,
			f"model_checkpoints/{embedding_model_identifier}/downstream/last_losses.svg",
			f"Embedding model downstream task loss"
		)

	embedding_downstream_dict = {
		'embedding_model_state_dict': embedding_model.state_dict(),
		'downstream_model_state_dict': downstream_model.state_dict(),
		'losses': losses
	}

	path = f"model_checkpoints/{embedding_model_identifier}/downstream/model_downstream_dict.pth"
	torch.save(embedding_downstream_dict, path)