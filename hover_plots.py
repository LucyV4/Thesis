import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import pyperclip

import torch

from typing import List, Tuple

import numpy as np
import seaborn as sns

from umap import UMAP

from models import TL_FFN_REG as MODEL_FILE
from data_loader import DataLoader

device = torch.device("cpu")

# Loads data
dataloader = DataLoader(
	data_cols=["avg_amplitude","max_amplitude","mean_percycle_max_speed","mean_percycle_avg_speed","mean_tapping_interval","amp_slope","speed_slope","cov_tapping_interval","cov_amp","cov_per_cycle_speed_maxima","cov_per_cycle_speed_avg","num_interruptions2"],
	label_cols=["ids", "UPDRS", "visit", "on_medication", "hand"]
)

# Loads model
epoch = 7
main_label = "UPDRS"
model_option_dict = {
	"learning_rate": 0.00001,
	"dropout": 0.1,
	"margin": 1,
	"l2": 0.01,
	"batch_size": 100000,
	"n_neighbours": 5,
	"min_dist": 0.25,
}

model = MODEL_FILE.Model(model_option_dict["dropout"])
model_identifier = MODEL_FILE.get_identifier(
		main_label,
		model_option_dict["l2"],
		model_option_dict["dropout"],
		model_option_dict["learning_rate"],
		model_option_dict["margin"],
		model_option_dict["batch_size"],
		model_option_dict["n_neighbours"],
		model_option_dict["min_dist"]
)

checkpoint = torch.load(f"model_checkpoints/{model_identifier}/TL_FFN_epoch{epoch}.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Gets embedded data
X_train_raw = dataloader.get_train_data()
X_test_raw = dataloader.get_test_data()
X_holdout_raw = dataloader.get_holdout_data()

y_train = dataloader.get_train_labels()
y_test = dataloader.get_test_labels()
y_holdout = dataloader.get_holdout_labels()

X = torch.tensor(X_train_raw, dtype=torch.float)
X_out: np.ndarray = model(X).cpu().detach().numpy()
umap: UMAP = UMAP(n_neighbors=model_option_dict["n_neighbours"], min_dist=model_option_dict["min_dist"], random_state=73, n_jobs=1).fit(X_out)

X_train = model(torch.tensor(X_train_raw, dtype=torch.float)).cpu().detach().numpy()
X_test = model(torch.tensor(X_test_raw, dtype=torch.float)).cpu().detach().numpy()
X_holdout = model(torch.tensor(X_holdout_raw, dtype=torch.float)).cpu().detach().numpy()

train_embedding = umap.transform(X_train)
test_embedding = umap.transform(X_test)
holdout_embedding = umap.transform(X_holdout)

print("Embedded data")
print("---------------------------------------")

# Calculates colors
color_col = dataloader.label_cols.index(main_label)
label_set = np.unique(y_train[:, color_col])
colors = sns.color_palette("hls", len(label_set)+1)

color_dict = {}
for l, c in zip(label_set, colors):
	color_dict[l] = c

(fig, ax) = plt.subplots(1, 3, figsize=(60,15))
ax: List[matplotlib.axes.Axes] = ax

# Makes plots
scatter_train = ax[0].scatter(
	train_embedding[:,0],
	train_embedding[:,1],
	c=[color_dict[l] for l in y_train[:, color_col]],
	alpha=0.7,
	zorder=1000
)

ax[0].set_title("Training embedding")

scatter_test = ax[1].scatter(
	test_embedding[:,0],
	test_embedding[:,1],
	c=[color_dict[l] for l in y_test[:, color_col]],
	alpha=0.7,
	zorder=1000
)

ax[1].set_title("Test embedding")

scatter_holdout = ax[2].scatter(
	holdout_embedding[:,0],
	holdout_embedding[:,1],
	c=[color_dict[l] for l in y_holdout[:, color_col]],
	alpha=0.7,
	zorder=1000
)

ax[2].set_title("Holdout embedding")

fig.suptitle(f"Interactive embeddings of ModelV2\nModel params: {main_label} - L2: {model_option_dict['l2']} - Dropout: {model_option_dict['dropout']} - LR: {model_option_dict['learning_rate']} - Margin: {model_option_dict['margin']}\nUMAP parameters: Neighbours: {model_option_dict['n_neighbours']} - Min dist: {model_option_dict['min_dist']}")

for axs in ax:
    axs.set_xlabel("UMAP axis 1")
    axs.set_ylabel("UMAP axis 2")

# Cool code that adds hovering to plots
# FROM: https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
annots = [axs.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"), fontsize=6) for axs in ax]
for annot in annots: 
    annot.set_visible(False)
    annot.set_zorder(10000)
    annot.set_clip_on(False)

def update_train_annot(event, ind):
	pos = (event.xdata, event.ydata)
	points = np.array([[x,y] for x,y in [train_embedding[ix, :] for ix in ind]])
	dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
	closest = np.argmin(dists)
	pat_id = ind[closest]
	pos = points[closest]

	annots[0].xy = pos
	text = f"Patient: {y_train[pat_id, 0]}\nUPDRS: {y_train[pat_id, 1]}\nVisit: {y_train[pat_id, 2]}\nMeds: {y_train[pat_id, 3]}\nHand: {y_train[pat_id, 4]}"
	annots[0].set_text(text)
	annots[0].get_bbox_patch().set_facecolor(color_dict[y_train[pat_id, color_col]])
	annots[0].get_bbox_patch().set_alpha(0.4)

def update_test_annot(event, ind):
	pos = (event.xdata, event.ydata)
	points = np.array([[x,y] for x,y in [test_embedding[ix, :] for ix in ind]])
	dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
	closest = np.argmin(dists)
	pat_id = ind[closest]
	pos = points[closest]

	annots[1].xy = pos
	text = f"Patient: {y_test[pat_id, 0]}\nUPDRS: {y_test[pat_id, 1]}\nVisit: {y_test[pat_id, 2]}\nMeds: {y_test[pat_id, 3]}\nHand: {y_test[pat_id, 4]}"
	annots[1].set_text(text)
	annots[1].get_bbox_patch().set_facecolor(color_dict[y_test[pat_id, color_col]])
	annots[1].get_bbox_patch().set_alpha(0.4)
 
def update_holdout_annot(event, ind):
	pos = (event.xdata, event.ydata)
	points = np.array([[x,y] for x,y in [holdout_embedding[ix, :] for ix in ind]])
	dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
	closest = np.argmin(dists)
	pat_id = ind[closest]
	pos = points[closest]

	annots[2].xy = pos
	text = f"Patient: {y_holdout[pat_id, 0]}\nUPDRS: {y_holdout[pat_id, 1]}\nVisit: {y_holdout[pat_id, 2]}\nMeds: {y_holdout[pat_id, 3]}\nHand: {y_holdout[pat_id, 4]}"
	annots[2].set_text(text)
	annots[2].get_bbox_patch().set_facecolor(color_dict[y_holdout[pat_id, color_col]])
	annots[2].get_bbox_patch().set_alpha(0.4)

def hover(event):
	vis = [annot.get_visible() for annot in annots]
	if event.inaxes == ax[0]:
		for annot in annots:
			if annot != annots[0]: annot.set_visible(False)
		cont, ind = scatter_train.contains(event)
		if cont:
			update_train_annot(event, ind["ind"])
			annots[0].set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis[0]:
				annots[0].set_visible(False)
				fig.canvas.draw_idle()
	if event.inaxes == ax[1]:
		for annot in annots:
			if annot != annots[1]: annot.set_visible(False)
		cont, ind = scatter_test.contains(event)
		if cont:
			update_test_annot(event, ind["ind"])
			annots[1].set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis[1]:
				annots[1].set_visible(False)
				fig.canvas.draw_idle()
	if event.inaxes == ax[2]:
		for annot in annots:
			if annot != annots[2]: annot.set_visible(False)
		cont, ind = scatter_holdout.contains(event)
		if cont:
			update_holdout_annot(event, ind["ind"])
			annots[2].set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis[2]:
				annots[2].set_visible(False)
				fig.canvas.draw_idle()

def click(event):
	if event.inaxes == ax[0]:
		cont, ind = scatter_train.contains(event)
		if cont:
			pos = (event.xdata, event.ydata)
			points = np.array([[x,y] for x,y in [train_embedding[ix, :] for ix in ind["ind"]]])
			dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
			closest = np.argmin(dists)
			pat_id = ind["ind"][closest]
			text = f"Train: {y_train[pat_id, 0]} - UPDRS {y_train[pat_id, 1]} - Visit {y_train[pat_id, 2]} - { 'On Meds' if y_train[pat_id, 3] == 1 else 'Off Meds' } - Hand {y_train[pat_id, 4]}"
			pyperclip.copy(text)
	elif event.inaxes == ax[1]:
		cont, ind = scatter_test.contains(event)
		if cont:
			pos = (event.xdata, event.ydata)
			points = np.array([[x,y] for x,y in [test_embedding[ix, :] for ix in ind["ind"]]])
			dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
			closest = np.argmin(dists)
			pat_id = ind["ind"][closest]
			text = f"Test: {y_test[pat_id, 0]} - UPDRS {y_test[pat_id, 1]} - Visit {y_test[pat_id, 2]} - { 'On Meds' if y_test[pat_id, 3] == 1 else 'Off Meds' } - Hand {y_test[pat_id, 4]}"
			pyperclip.copy(text)
	elif event.inaxes == ax[2]:
		cont, ind = scatter_holdout.contains(event)
		if cont:
			pos = (event.xdata, event.ydata)
			points = np.array([[x,y] for x,y in [holdout_embedding[ix, :] for ix in ind["ind"]]])
			dists = np.hypot(points[:,0]-pos[0], points[:,1]-pos[1])
			closest = np.argmin(dists)
			pat_id = ind["ind"][closest]
			text = f"Holdout: {y_holdout[pat_id, 0]} - UPDRS {y_holdout[pat_id, 1]} - Visit {y_holdout[pat_id, 2]} - { 'On Meds' if y_holdout[pat_id, 3] == 1 else 'Off Meds' } - Hand {y_holdout[pat_id, 4]}"
			pyperclip.copy(text)

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("button_press_event", click)

plt.show()