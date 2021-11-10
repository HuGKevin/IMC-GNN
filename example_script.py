import datasets as ds
import models as md
import lopocv_trainer as lopo

import numpy as np
import pandas as pd

import torch
import copy
from torch_geometric.data import DataLoader
from random import randint
from collections import Counter
import sys
import re

# Setup some variables
save_dir = '/home/kevin/project/NSCLC_final/model_saves/'

# Load datasets
dataset = ds.NSCLC_Dataset(dataset = 'c2', neighbordef = 'naive', subgraph = 'windows', naive_radius = 25, width = 50, window_num = 100)

# Create model
model = md.GCN_Train(33, 2, 16, metric = 'accuracy', es_min_iter=35)

# Instantiate LOPOCV object
lopo_trainer = lopo.LOPOCV(model = model, dataset = dataset, devices = 'cuda:0', batch_size = 128)

lopo_auc_track = []
for i in range(0, 35):
    # Train an epoch
    lopo_trainer.train(save_dir = save_dir, verbose = True)

    # Validate
    lopo_auc_track.append(lopo_trainer.validate(device = 'cuda:0', verbose = False, aggregate_func = 'majority_vote'))

# Pull out all the statistics
for patient in lopo_trainer.patient_list:
    tracker = pd.DataFrame(data = {'training':lopo_trainer.model_trainers[patient].train_acc,
                                   'validation':lopo_trainer.model_trainers[patient].flag.auc_track})

    tracker.to_csv(save_dir + 'lopo' + patient + 'tracker.csv')
    
final_auc = pd.DataFrame(data = {'epoch':range(0,35),
                                 'auc':lopo_auc_track})
final_auc.to_csv(save_dir + 'overall_auc.csv')

# Visualize results?




