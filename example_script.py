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

# Load datasets
dataset = ds.NSCLC_Dataset(dataset = 'c2', neighbordef = 'naive', subgraph = 'windows', naive_radius = 25, width = 50, window_num = 100)

# Create model
model = md.GCN_Train(33, 2, 16, metric = 'accuracy', es_min_iter=35)

# Instantiate LOPOCV object
test = lopo.LOPOCV(model = model, dataset = dataset, devices = 'cuda:0', batch_size = 32)

# Train an epoch
test.train(save_dir = '/home/kevin/project/NSCLC_final/model_saves/', verbose = True)

# Validate
test.load_all_models_from_dir('/home/kevin/project/NSCLC_final/model_saves/')
test.validate(device = 'cuda:0', verbose = True)

# Loading models for backup
# direc = '/home/kevin/project/NSCLC_final/model_saves/'
# file = direc + 'lopo' + str(patient2) + '.mdl'
test.load_all_models_from_dir('/home/kevin/project/NSCLC_final/model_saves/')

# Visualize everything





