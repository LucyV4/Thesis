#!/bin/sh

# This file is all the models we trained from the bigger dataset onward, we kept track so we could rerun these scripts to create the visualizations again
# The commented lines are the ones not used in the thesis but tried to find better model parameters

# Training from bigger dataset onward
# python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=19 l2=0.1 dropout=0.1 batch_size=100000 triplet_label=UPDRS
python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=16 l2=0.01 dropout=0.1 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=25 l2=0.1 dropout=0.5 batch_size=100000 triplet_label=UPDRS
python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=25 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS

# python main.py learning_rate=0.00001 margin=3 n_neighbours=5 n_epoch=25 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.000001 margin=3 n_neighbours=5 n_epoch=25 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=25 l2=0.05 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.00001 margin=1 n_neighbours=5 n_epoch=22 l2=0.005 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=25 l2=0.005 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python main.py learning_rate=0.000001 margin=3 n_neighbours=5 n_epoch=25 l2=0.005 dropout=0.5 batch_size=100000 triplet_label=UPDRS

python3 main_small.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=28 l2=0.01 dropout=0.1 batch_size=100000 triplet_label=UPDRS
python3 main_small.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=50 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python3 main_small.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=30 l2=0.001 dropout=0.5 batch_size=100000 triplet_label=UPDRS

# This did well so we trained this for longer
python main.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=94 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS

# Test going back to ids for making triplets, not training well enough
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=150 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=ids

# Work well
python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=57 l2=0.01 dropout=0.1 batch_size=100000 triplet_label=UPDRS
python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=44 l2=0.01 dropout=0.3 batch_size=100000 triplet_label=UPDRS
python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=43 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=52 l2=0.01 dropout=0.7 batch_size=100000 triplet_label=UPDRS

# Testing around best params, did not improve
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=40 l2=0.1 dropout=0.1 batch_size=100000 triplet_label=UPDRS
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=40 l2=0.005 dropout=0.1 batch_size=100000 triplet_label=UPDRS
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=38 l2=0.1 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python mainV2.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=45 l2=0.005 dropout=0.5 batch_size=100000 triplet_label=UPDRS

# V2 with 12 dimensions
python mainV2_12DIM.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=50 l2=0.01 dropout=0.1 batch_size=100000 triplet_label=UPDRS
# python mainV2_12DIM.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=33 l2=0.01 dropout=0.3 batch_size=100000 triplet_label=UPDRS
# python mainV2_12DIM.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=68 l2=0.01 dropout=0.5 batch_size=100000 triplet_label=UPDRS
# python mainV2_12DIM.py learning_rate=0.000001 margin=1 n_neighbours=5 n_epoch=17 l2=0.01 dropout=0.7 batch_size=100000 triplet_label=UPDRS