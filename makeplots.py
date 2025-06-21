import torch

from models.TL_FFN_REG import Model

from data_loader import DataLoader
from model_data_visualizer import ModelDataVisualizer
from triplet_selector import TripletSelector

MODEL_IDENTIFIER = "big_data_TL_FFN_REG_0.01L2_0.1D_1e-05LR_3Margin_1000BatchSize_3Neighbours_0.1MinDist"

EPOCHS = 50

model = Model()

data_loader = DataLoader(
    data_cols=["avg_amplitude","max_amplitude","mean_percycle_max_speed","mean_percycle_avg_speed","mean_tapping_interval","amp_slope","speed_slope","cov_tapping_interval","cov_amp","cov_per_cycle_speed_maxima","cov_per_cycle_speed_avg","num_interruptions2"],
    label_cols=["ids", "UPDRS", "visit", "on_medication", "hand"]
)

X_train = data_loader.get_train_data()
y_train = data_loader.get_train_labels()
X_test = data_loader.get_test_data()
y_test = data_loader.get_test_labels()

triplet_selector = TripletSelector("data/triplets.pkl", data_loader.label_cols)

for epoch in range(EPOCHS):
    path = f"model_checkpoints/{MODEL_IDENTIFIER}/TL_FFN_epoch{epoch+1}.pth"
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    visualizer = ModelDataVisualizer(
        model,
        "cpu",
        data_loader.label_cols,
        "ids",
        n_neighbours=5,
        min_dist=0.5
    )

    visualizer.fit(X_train)
    visualizer.visualize(X_train, y_train, f"UMAP - Epoch {epoch+1} - Train data", f"model_figures/{MODEL_IDENTIFIER}/TL_FFN_umap_epoch{epoch+1}_train.svg")
    visualizer.visualize(X_test, y_test, f"UMAP - Epoch {epoch+1} - Test data", f"model_figures/{MODEL_IDENTIFIER}/TL_FFN_umap_epoch{epoch+1}_test.svg")
    
    print(f"Epoch: {epoch+1}/{EPOCHS}")