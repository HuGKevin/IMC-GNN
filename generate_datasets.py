import sys
sys.path.append('/home/kevin/project')

from IMC_GNN import datasets as ds
from IMC_GNN import models as md
import torch
from torch_geometric.data import DataLoader

window_widths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
naive_radii = [20, 25, 30, 40, 50]

# Generate all necessary windows for c1 of NSCLC data.
for width in window_widths:
    for radius in naive_radii:
        dataset = ds.NSCLC_Dataset(dataset = 'c1', neighbordef = 'naive', subgraph = 'windows', naive_radius = radius, width = width, window_num = 100)
        print(f'Finished processing for c1, width {width}, radius {radius}.')

# Generate all necessary windows for c2 of NSCLC data.
for width in window_widths:
    for radius in naive_radii:
        dataset = ds.NSCLC_Dataset(dataset = 'c2', neighbordef = 'naive', subgraph = 'windows', naive_radius = radius, width = width, window_num = 100)
        print(f'Finished processing for c2, width {width}, radius {radius}.')
