import torch.nn as nn
import torch

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from umap import UMAP

RANDOM_NUM = 73

class ModelDataVisualizer:
	def __init__(self, model: nn.Module, device: str, labels: list, main_label: str, n_neighbours: int = 15, min_dist: float = 0.1):
		self.n_neighbours: int = n_neighbours
		self.min_dist: int = min_dist
		self.model: nn.Module = model
		self.device: str = device

		self.label_names = labels
		self.color_col = labels.index(main_label)
		temp_cols = np.array(range(len(labels)))
		self.char_cols = temp_cols[(temp_cols != self.color_col)]  

	def fit(self, data: np.ndarray):
		X = torch.tensor(data, dtype=torch.float).to(self.device)
		X_out: np.ndarray = self.model(X).cpu().detach().numpy()
		self.model_umap: UMAP = UMAP(n_neighbors=self.n_neighbours, min_dist=self.min_dist, random_state=RANDOM_NUM, n_jobs=1).fit(X_out)
		self.raw_umap: UMAP = UMAP(n_neighbors=self.n_neighbours, min_dist=self.min_dist, random_state=RANDOM_NUM, n_jobs=1).fit(data)

	def calc_colors(self, labels: np.ndarray) -> dict:
		label_set = np.unique(labels)
		
		colors = []
		if len(label_set) < 10:
			colors = sns.color_palette("hls", len(label_set)+1)
		elif len(label_set) <= 40:
			colors = np.array([
				sns.color_palette("dark"),
				sns.color_palette("bright"),
				sns.color_palette("tab10"),
				sns.color_palette("pastel")
			], dtype=float).T.reshape((3,40)).T
		elif len(label_set) <= 60:
			colors = [*plt.cm.tab20(np.linspace(0,1,20)), *plt.cm.tab20b(np.linspace(0,1,20)), *plt.cm.tab20c(np.linspace(0,1,20))]
		else:
			for color1, color2 in zip(sns.color_palette("hls", math.ceil(len(label_set)/2)), sns.color_palette("husl", math.ceil(len(label_set)/2))):
				colors.append(color1)
				colors.append(color2)
		
		color_dict = {}
		for l, c in zip(label_set, colors):
			color_dict[l] = c
		return color_dict

	def calc_markers(self, labels: np.ndarray) -> dict:
		label_set = np.unique(labels)
		markers = (["o", "X", "s", "D", "h", "8", "1"])[:len(label_set)]
		marker_dict = {}
		for l, m in zip(label_set, markers):
			marker_dict[l] = m
		return marker_dict

	def visualize_small_model(self, data: np.ndarray, labels: np.ndarray, fig_title: str, filename: str = None, subtitle_str: str = ""):
		X = torch.tensor(data, dtype=torch.float).to(self.device)
		embedding: np.ndarray = self.model(X).cpu().detach().numpy()

		colors = self.calc_colors(labels[:,self.color_col])
		legend_elements = [Line2D([],[], marker="o", color=c, label=l, linestyle="None") for l,c in colors.items()]

		# Creates a figure and makes space for the color legend
		plt.clf()
		
		fig = plt.figure(figsize=(60,15))
		ax = fig.add_subplot(1, 4, 1, projection='3d')

		# First single figure without any other label markers
		ax.scatter(
			embedding[:,0],
			embedding[:,1],
			embedding[:,2],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		ax.set_xlabel("First model embedding dimension")
		ax.set_ylabel("Second model embedding dimension")
		ax.set_zlabel("Third model embedding dimension")
		ax.legend(handles=legend_elements)

		ax = fig.add_subplot(1, 4, 2)
		ax.scatter(
			embedding[:,0],
			embedding[:,1],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		ax.set_xlabel("First model embedding dimension")
		ax.set_ylabel("Second model embedding dimension")
		ax.legend(handles=legend_elements)
		
		ax = fig.add_subplot(1, 4, 3)
		ax.scatter(
			embedding[:,0],
			embedding[:,2],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		ax.set_xlabel("First model embedding dimension")
		ax.set_ylabel("Third model embedding dimension")
		ax.legend(handles=legend_elements)
		
		ax = fig.add_subplot(1, 4, 4)
		ax.scatter(
			embedding[:,1],
			embedding[:,2],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		ax.set_xlabel("Second model embedding dimension")
		ax.set_ylabel("Third model embedding dimension")
		ax.legend(handles=legend_elements)
		
		plt.suptitle(f"{fig_title}\n{subtitle_str}", fontsize=28)

		if filename: 
			plt.savefig(fname=filename)
			plt.close("all")
		else: plt.show()

	def UMAP_visualize_single(self, data, labels, fig_title: str, filename: str = None, subtitle_str: str = ""):
		X = torch.tensor(data, dtype=torch.float).to(self.device)
		X_out: np.ndarray = self.model(X).cpu().detach().numpy()
		embedding = self.model_umap.transform(X_out)

		colors = self.calc_colors(labels[:,self.color_col])

		plt.clf()
		fig = plt.figure(figsize=(15,15))

		axs = plt.subplot(1, 1, 1)
		axs.scatter(
			embedding[:,0],
			embedding[:,1],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		axs.set_xlabel("UMAP axis 1", fontsize=14)
		axs.set_ylabel("UMAP axis 2", fontsize=14)

		axs_legend_handles = []
		for l, c in colors.items():
			axs_legend_handles.append(Line2D([],[], marker='o', color=c, label=f"{self.label_names[self.color_col]}: {l}", linestyle="None"))
		axs.legend(handles=axs_legend_handles, fontsize=14)

		plt.suptitle(fig_title, fontsize=20)
		axs.set_title(subtitle_str, fontsize=14)

		if filename: 
			plt.savefig(fname=filename)
			plt.close("all")
		else: plt.show()

	def UMAP_visualize(self, data: np.ndarray, labels: np.ndarray, fig_title: str, filename: str = None, subtitle_str: str = ""):
		X = torch.tensor(data, dtype=torch.float).to(self.device)
		X_out: np.ndarray = self.model(X).cpu().detach().numpy()
		embedding = self.model_umap.transform(X_out)

		colors = self.calc_colors(labels[:,self.color_col])

		# Creates a figure and makes space for the color legend
		plt.clf()
		plot_width = 15 * len(self.label_names)
		plot_height = 15
		fig = plt.figure(figsize=(plot_width,plot_height))

		# First single figure without any other label markers
		axs = plt.subplot(1,len(self.char_cols)+1, 1)
		axs.scatter(
			embedding[:,0],
			embedding[:,1],
			c=[colors[l] for l in labels[:, self.color_col]],
			alpha=0.7
		)
		axs.set_xlabel("UMAP axis 1", fontsize=14)
		axs.set_ylabel("UMAP axis 2", fontsize=14)

		axs_legend_handles = []
		for l, c in colors.items():
			axs_legend_handles.append(Line2D([],[], marker='o', color=c, label=f"{self.label_names[self.color_col]}: {l}", linestyle="None"))
		axs.legend(handles=axs_legend_handles, fontsize=14)

		# For each of the secondary labels adds markers to the figure
		# TO CHANGE
		for plot_i, label_i in enumerate(self.char_cols):
			# Sets the correct subplot position and makes the handles for the markers 
			axs = plt.subplot(1, len(self.label_names), plot_i+2)
			axs_legend_handles = []
   
			# Gets a dictionary for the markers
			# markers = self.calc_markers(labels[:,label_i])
			secondary_colors = self.calc_colors(labels[:,label_i])

			axs.scatter(
				embedding[:,0],
				embedding[:,1],
				c=[secondary_colors[l] for l in labels[:,label_i]],
				alpha=0.7
			)
			axs.set_xlabel("UMAP axis 1", fontsize=14)
			axs.set_ylabel("UMAP axis 2", fontsize=14)
   
			for l, c in secondary_colors.items():
				axs_legend_handles.append(Line2D([],[], marker='o', color=c, label=f"{self.label_names[label_i]}: {l}", linestyle="None"))
			axs.legend(handles=axs_legend_handles, fontsize=14)
		
		# Adds a title and the legend outside of the plot
		plt.suptitle(f"{fig_title}\n{subtitle_str}", fontsize=28)
		# legend_elements = [Line2D([],[], marker="o", color=c, label=l, linestyle="None") for l,c in colors.items()]
		# fig.legend(bbox_to_anchor=(((plot_width-3)/plot_width), 0.5), handles=legend_elements, loc="center left")

		if filename: 
			plt.savefig(fname=filename)
			plt.close("all")
		else: plt.show()
  
	def compare_visualize(self, data: np.ndarray, labels: np.ndarray, fig_title: str, filename: str, fig_subtitle: str = ""):
		X = torch.tensor(data, dtype=torch.float).to(self.device)
		X_out: np.ndarray = self.model(X).cpu().detach().numpy()
		model_emb = self.model_umap.transform(X_out)
		raw_emb = self.raw_umap.transform(data)
  
		fig, axs = plt.subplots(1, 2, figsize=(30,15))
  
		colors = self.calc_colors(labels[:,self.color_col])

		axs[0].set_xlabel("UMAP axis 1", fontsize=14)
		axs[0].set_ylabel("UMAP axis 2", fontsize=14)
		axs[0].set_title("Raw data visualization", fontsize=18)
		axs[0].scatter(
			raw_emb[:,0],
			raw_emb[:,1],
			c=[colors[l] for l in labels[:,self.color_col]],
			alpha=0.7
		)

		axs[1].set_xlabel("UMAP axis 1", fontsize=14)
		axs[1].set_ylabel("UMAP axis 2", fontsize=14)
		axs[1].set_title("Embedded data visualization", fontsize=18)
		axs[1].scatter(
			model_emb[:,0],
			model_emb[:,1],
			c=[colors[l] for l in labels[:,self.color_col]],
			alpha=0.7
		)
   
		axs_legend_handles = []
		for l, c in colors.items():
			axs_legend_handles.append(Line2D([],[], marker='o', color=c, label=f"{self.label_names[self.color_col]}: {l}", linestyle="None"))
		axs[0].legend(handles=axs_legend_handles, fontsize=14)
		axs[1].legend(handles=axs_legend_handles, fontsize=14)
  

		fig.suptitle(f"{fig_title}\n{fig_subtitle}", fontsize=24)
		plt.savefig(fname=filename)

	def overlay(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, fig_title: str, filename: str, fig_subtitle: str = ""):
		X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(self.device)
		X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(self.device)
		train_out: np.ndarray = self.model(X_train_tensor).cpu().detach().numpy()
		test_out: np.ndarray = self.model(X_test_tensor).cpu().detach().numpy()
		train_embedding = self.model_umap.transform(train_out)
		test_embedding = self.model_umap.transform(test_out)
  
		fig = plt.figure(figsize=(15,15))
		
		plt.xlabel("UMAP axis 1", fontsize=14)
		plt.ylabel("UMAP axis 2", fontsize=14)
		plt.scatter(
			train_embedding[:,0],
			train_embedding[:,1],
			alpha=0.7,
			label="Train embedding"
		)

		plt.scatter(
			test_embedding[:,0],
			test_embedding[:,1],
			alpha=0.7,
			label="Test embedding"
		)

		plt.legend(fontsize=14)
  
		fig.suptitle(f"{fig_title}\n{fig_subtitle}", fontsize=20)
		plt.savefig(fname=filename)

def visualize_loss(losses: np.ndarray, epochs: int, path: str, subtitle_str: str = ""):
	x = range(1, epochs+1)
	plt.clf()
	plt.plot(x, losses[:,0], label="Train loss")
	plt.plot(x, losses[:,1], label="Test loss")
	plt.xlabel("Epoch", fontsize=14)
	plt.title(subtitle_str, fontsize=20)
	plt.suptitle("Loss per epoch", fontsize=28)
	plt.ylabel("Loss", fontsize=14)
	plt.legend(fontsize=14)
	plt.savefig(path)