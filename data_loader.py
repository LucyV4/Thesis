import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

NUMPY_RANDOM_SEED = 73

class DataLoader: 
	def __init__(self, data_cols, label_cols):
		print("Loading data")
		self.orig_data = pd.read_csv("data/video_features.csv")
		self.extra_data = pd.read_csv("data/updated_video_features.csv")
  
		self.data = pd.concat([self.orig_data, self.extra_data], axis=0, ignore_index=True)

		self.data.rename(columns={"label":"UPDRS"}, inplace=True)
		
		self.data.dropna(inplace=True)
  
		# self.add_keypoint_data() # commented out as update video keypoints pkl file has not been updated

		self.data_cols = data_cols
		self.label_cols = label_cols
  
		self.extract_path_data()

		self.set_train_ids()

		self.scaler = StandardScaler().fit(self.data[self.data["ids"].isin(self.train_ids)][self.data_cols].to_numpy())

		print("Data loaded")
		print("---------------------------------------")

	def add_keypoint_data(self):
		keypoint_data = pd.read_pickle("data/video_keypoints.pkl")
		self.data["distances"] = keypoint_data["distances"]
		self.data["keypoints"] = keypoint_data["keypoints"]

	def extract_path_data(self):
		regex = r"Visit (\d+).*/(On|Off)_2(L|R)"
		extracted = []

		for path in self.data["video_path"].to_numpy():
			match = re.search(regex, path)
			extracted.append([
				int(match.group(1)),
				1 if match.group(2) == "On" else 0,
				1 if match.group(3) == "L" else 2
			])
		extracted = np.array(extracted)

		self.data["visit"] = extracted[:,0]
		self.data["on_medication"] = extracted[:,1]
		self.data["hand"] = extracted[:,2]

	def set_train_ids(self):
		np.random.seed(NUMPY_RANDOM_SEED)

		original_ids = np.unique(self.orig_data["ids"])
		orig_train_ids = np.random.choice(original_ids, size=50, replace=False)
		orig_test_ids = np.array([item for item in original_ids if item not in orig_train_ids])

		extra_ids = np.array([item for item in np.unique(self.extra_data["ids"]) if item not in original_ids]) # 114 new ids
		np.random.shuffle(extra_ids)
		extra_train_ids = extra_ids[:60]
		extra_test_ids = extra_ids[60:90]
		extra_holdout_ids = extra_ids[90:]
  
		self.train_ids = np.append(orig_train_ids, extra_train_ids)
		self.test_ids = np.append(orig_test_ids, extra_test_ids)
		self.holdout_ids = extra_holdout_ids.copy()

		print(f"{len(self.train_ids)} train patients")
		print(f"{len(self.test_ids)} test patients")
		print(f"{len(self.holdout_ids)} holdout patients")
		# Maybe for later???
		# self.train_indices = list(self.data[self.data["ids"].isin(self.train_ids)].index)
		# self.test_indices = list(self.data[~self.data["ids"].isin(self.train_ids)].index)

	def get_train_data(self) -> np.ndarray:
		X = self.data[self.data["ids"].isin(self.train_ids)][self.data_cols].to_numpy()
		return self.scaler.transform(X, True)

	def get_test_data(self) -> np.ndarray:
		X = self.data[self.data["ids"].isin(self.test_ids)][self.data_cols].to_numpy()
		return self.scaler.transform(X, True)

	def get_holdout_data(self) -> np.ndarray:
		X = self.data[self.data["ids"].isin(self.holdout_ids)][self.data_cols].to_numpy()
		return self.scaler.transform(X, True)
  
	def get_train_labels(self) -> np.ndarray:
		return self.data[self.data["ids"].isin(self.train_ids)][self.label_cols].to_numpy()
  
	def get_test_labels(self) -> np.ndarray:
		return self.data[self.data["ids"].isin(self.test_ids)][self.label_cols].to_numpy()

	def get_holdout_labels(self) -> np.ndarray:
		return self.data[self.data["ids"].isin(self.holdout_ids)][self.label_cols].to_numpy()