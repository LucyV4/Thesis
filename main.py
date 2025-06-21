import torch
import sys

from data_loader import DataLoader
from models import TL_FFN_REG

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

if __name__ == "__main__":
	options = {}
 
	ints = ["margin", "n_neighbours", "n_epoch", "batch_size", "start_epoch"]
	floats = ["learning_rate", "l2", "dropout"]
 
	for arg in sys.argv[1:]:
		key, value = (arg.split("="))
		if key in ints: options[key] = int(value)
		elif key in floats: options[key] = float(value)
		else: options[key] = value

	data_loader = DataLoader(
		data_cols=["avg_amplitude","max_amplitude","mean_percycle_max_speed","mean_percycle_avg_speed","mean_tapping_interval","amp_slope","speed_slope","cov_tapping_interval","cov_amp","cov_per_cycle_speed_maxima","cov_per_cycle_speed_avg","num_interruptions2"],
		label_cols=["UPDRS", "visit", "on_medication", "hand"]
	)

	TL_FFN_REG.train(data_loader, device, options)
	TL_FFN_REG.visualize(data_loader, device, options)