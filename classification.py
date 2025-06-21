import torch

import pickle
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models import TL_FFN_REG as MODEL_FILE
from models import downstream_classifier, direct_classifier
from data_loader import DataLoader

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

direct = False

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

names = [
    "Naive Bayes",
    "QDA",
    "Nearest Neighbors",
    "Gaussian Process",
    "RBF SVM",
]

downstream_classifiers = [
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(5),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    SVC(gamma=2, C=1, random_state=42, probability=True),
]

direct_classifiers = [
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    SVC(gamma=2, C=1, random_state=42, probability=True),
]

def fit_downstream_classifiers(option_dict: dict, dataloader: DataLoader):
	classification_label = option_dict.get("classification_label")
	embedding_model_options: dict = option_dict.get("embedding_model")

	embedding_model_epoch = embedding_model_options.get("epoch")
	embedding_model_identifier = MODEL_FILE.get_identifier(
		classification_label,
		embedding_model_options["l2"],
		embedding_model_options["dropout"],
		embedding_model_options["lr"],
		embedding_model_options["margin"],
		embedding_model_options["batch_size"],
		embedding_model_options["n_neighbours"],
		embedding_model_options["min_dist"]
	)

	label_index = dataloader.label_cols.index(classification_label)

	model = MODEL_FILE.Model(dropout=embedding_model_options["dropout"]).to(device)
	checkpoint = torch.load(f"model_checkpoints/{embedding_model_identifier}/TL_FFN_epoch{embedding_model_epoch}.pth", map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	X_train = torch.tensor(dataloader.get_train_data(), dtype=torch.float).to(device)
	X_train: np.ndarray = model(X_train).cpu().detach().numpy()
	y_train = np.array(dataloader.get_train_labels()[:, label_index], dtype=np.uint8)

	for i, classifier in tqdm(enumerate(downstream_classifiers), desc="classifier", total=len(downstream_classifiers)): downstream_classifiers[i] = (classifier.fit(X_train, y_train))

def fit_direct_classifiers(option_dict: dict, dataloader: DataLoader):
	classification_label = option_dict.get("classification_label")
	label_index = dataloader.label_cols.index(classification_label)
    
	X_train = dataloader.get_train_data()
	y_train = np.array(dataloader.get_train_labels()[:, label_index], dtype=np.uint8)

	for i, classifier in tqdm(enumerate(direct_classifiers), desc="classifier", total=len(direct_classifiers)): direct_classifiers[i] = (classifier.fit(X_train, y_train))

def train_downstream_task(option_dict: dict, dataloader: DataLoader):
	classification_label = option_dict.get("classification_label")

	embedding_model_options: dict = option_dict.get("embedding_model")

	embedding_model_epoch = embedding_model_options.get("epoch")
	embedding_model_identifier = MODEL_FILE.get_identifier(
		classification_label,
		embedding_model_options["l2"],
		embedding_model_options["dropout"],
		embedding_model_options["lr"],
		embedding_model_options["margin"],
		embedding_model_options["batch_size"],
		embedding_model_options["n_neighbours"],
		embedding_model_options["min_dist"]
	)

	model = MODEL_FILE.Model(dropout=0.5).to(device)
	checkpoint = torch.load(f"model_checkpoints/{embedding_model_identifier}/TL_FFN_epoch{embedding_model_epoch}.pth", map_location=device, weights_only=False)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	downstream_classifier.train(model, device, dataloader, embedding_model_identifier, option_dict)

def train_direct_task(option_dict: dict, dataloader: DataLoader):
	direct_classifier.train(device, dataloader, option_dict)

def calculate_downstream_outputs(option_dict: dict, dataloader: DataLoader):
	classification_label = option_dict.get("classification_label")

	embedding_model_options = option_dict.get("embedding_model")

	embedding_model_identifier = MODEL_FILE.get_identifier(
		classification_label,
		embedding_model_options["l2"],
		embedding_model_options["dropout"],
		embedding_model_options["lr"],
		embedding_model_options["margin"],
		embedding_model_options["batch_size"],
		embedding_model_options["n_neighbours"],
		embedding_model_options["min_dist"]
	)

	dicts = torch.load(f"model_checkpoints/{embedding_model_identifier}/downstream/model_downstream_dict.pth", map_location=device, weights_only=False)

	out_size, in_size = dicts['downstream_model_state_dict']["l1.weight"].shape

	embedding_model = MODEL_FILE.Model(dropout=embedding_model_options["dropout"]).to(device)
	downstream_model = downstream_classifier.DownstreamClasisfier(in_size, out_size).to(device)

	embedding_model.load_state_dict(dicts['embedding_model_state_dict'])
	downstream_model.load_state_dict(dicts['downstream_model_state_dict'])
	embedding_model.eval()
	downstream_model.eval()

	X_train = embedding_model(torch.tensor(dataloader.get_train_data(), dtype=torch.float)).cpu().detach().numpy()
	X_test = embedding_model(torch.tensor(dataloader.get_test_data(), dtype=torch.float)).cpu().detach().numpy()
	X_holdout = embedding_model(torch.tensor(dataloader.get_holdout_data(), dtype=torch.float)).cpu().detach().numpy()

	train_output = {}
	test_output = {}
	holdout_output = {}

	train_output["self_nn"] = torch.nn.functional.softmax(downstream_model(torch.tensor(X_train, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()
	test_output["self_nn"] = torch.nn.functional.softmax(downstream_model(torch.tensor(X_test, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()
	holdout_output["self_nn"] = torch.nn.functional.softmax(downstream_model(torch.tensor(X_holdout, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()

	for classifier_name, fitted_classifier in zip(names, downstream_classifiers):
		train_output[classifier_name] = fitted_classifier.predict_proba(X_train)
		test_output[classifier_name] = fitted_classifier.predict_proba(X_test)
		holdout_output[classifier_name] = fitted_classifier.predict_proba(X_holdout)

	return train_output, test_output, holdout_output

def calculate_direct_outputs(option_dict: dict, dataloader: DataLoader):
	classification_label: str = option_dict.get("classification_label")
	l2 = option_dict.get("l2", 0.01)
	dropout = option_dict.get("dropout", 0.5)
	lr = option_dict.get("learning_rate", 0.0001)
	batch_size = option_dict.get("batch_size", 1000)

	identifier = direct_classifier.get_identifier(classification_label, l2, dropout, lr, batch_size)

	model_epoch = 150
	dicts = torch.load(f"model_checkpoints/{identifier}/model_direct_dict_epoch{model_epoch}.pth", map_location=device, weights_only=False)

	out_size, in_size = dicts['model_state_dict']["l5.weight"].shape

	direct_model = direct_classifier.DirectClasisfier(dropout, out_size).to(device)
	direct_model.load_state_dict(dicts['model_state_dict'])
	direct_model.eval()

	X_train = dataloader.get_train_data()
	X_test = dataloader.get_test_data()
	X_holdout = dataloader.get_holdout_data()

	train_output = {}
	test_output = {}
	holdout_output = {}

	train_output["self_nn"] = torch.nn.functional.softmax(direct_model(torch.tensor(X_train, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()
	test_output["self_nn"] = torch.nn.functional.softmax(direct_model(torch.tensor(X_test, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()
	holdout_output["self_nn"] = torch.nn.functional.softmax(direct_model(torch.tensor(X_holdout, dtype=torch.float).to(device)), dim=1).cpu().detach().numpy()

	for classifier_name, fitted_classifier in zip(names, direct_classifiers):
		train_output[classifier_name] = fitted_classifier.predict_proba(X_train)
		test_output[classifier_name] = fitted_classifier.predict_proba(X_test)
		holdout_output[classifier_name] = fitted_classifier.predict_proba(X_holdout)

	return train_output, test_output, holdout_output

if __name__ == "__main__":
	dataloader = DataLoader(
		data_cols=["avg_amplitude","max_amplitude","mean_percycle_max_speed","mean_percycle_avg_speed","mean_tapping_interval","amp_slope","speed_slope","cov_tapping_interval","cov_amp","cov_per_cycle_speed_maxima","cov_per_cycle_speed_avg","num_interruptions2"],
		label_cols=["ids", "UPDRS", "visit", "on_medication", "hand"]
	)

	direct_task_option_dict = {
		"classification_label": "UPDRS",
		"learning_rate": 0.0001,
		"dropout": 0.5,
		"l2": 0.01,
		"n_epoch": 150,
		"batch_size": 16,
		"weights": [0.6, 0.2, 0.1, 0.1, 1.5],
	}

	downstream_task_option_dict = {
		"classification_label": "UPDRS",
		"learning_rate": 0.0001,
		"l2": 0.01,
		"dropout": 0.5,
		"n_epoch": 300,
		"batch_size": 16,
		"weights": [0.6, 0.2, 0.1, 0.1, 1.5],
		"embedding_model": {
			"lr": 1e-5,
			"l2": 0.01,
			"dropout": 0.1,
			"margin": 1,
			"n_neighbours": 5,
			"min_dist": 0.25,
			"batch_size": 100000,
			"epoch": 7,
		}
	}

	# Training self-made classifier
	print("Training model classifiers")
	if direct: train_direct_task(direct_task_option_dict, dataloader)
	train_downstream_task(downstream_task_option_dict, dataloader)
	print("Done training model classifiers")
	print("----------------------------------------")
	
	# Fitting sklearn classifiers
	print("Fitting sklearn classifiers")
	if direct: fit_direct_classifiers(direct_task_option_dict, dataloader)
	fit_downstream_classifiers(downstream_task_option_dict, dataloader)
	print("Done fitting sklearn classifiers")
	print("----------------------------------------")


	# Calculating outputs
	print("Calculating outputs")
	if direct: direct_train_out, direct_test_out, direct_holdout_out = calculate_direct_outputs(direct_task_option_dict, dataloader)
	downstream_train_out, downstream_test_out, downstream_holdout_out = calculate_downstream_outputs(downstream_task_option_dict, dataloader)
	print("Done calculating outputs")
	print("----------------------------------------")

	embedding_model_options = downstream_task_option_dict["embedding_model"]
	classification_label = downstream_task_option_dict["classification_label"]
	embedding_model_identifier = MODEL_FILE.get_identifier(
		classification_label,
		embedding_model_options["l2"],
		embedding_model_options["dropout"],
		embedding_model_options["lr"],
		embedding_model_options["margin"],
		embedding_model_options["batch_size"],
		embedding_model_options["n_neighbours"],
		embedding_model_options["min_dist"]
	)
 
	if direct: 
		direct_result_dict = {
			"train": direct_train_out,
			"test": direct_test_out,
			"holdout": direct_holdout_out,
		}

		with open('classifier_results/direct.pkl', 'wb') as file:
			pickle.dump(direct_result_dict, file)
 
	downstream_result_dict = {
		"train": downstream_train_out,
		"test": downstream_test_out,
		"holdout": downstream_holdout_out,
	}

	with open(f'classifier_results/{embedding_model_identifier}.pkl', 'wb') as file:
		pickle.dump(downstream_result_dict, file)
