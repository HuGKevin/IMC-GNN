# We'll create a few different definitions of neighbor and see how that affects Fiedler values/vectors, i.e. giving us a greater sense of the connectivity of the graph. 

# Possible definitions
# 1. As defined in the data
# 2. All cells within a fixed radius of a center cell
# 3. Neighbors if the edges of the ellipses are within a certain distance of each other

from networkx.generators import directed
import pandas
from os import listdir
from os.path import isfile, join
import torch
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils.convert import to_networkx

# Load in all the data
folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
labels = ['DCB', 'NDB']
dataset = []

for label in labels:
    folderpath = join(folder, label)
    for file in listdir(folderpath):
        pointer = pandas.read_csv(join(folderpath, file)) # Read in the data
        # Construct adjacency matrix
        file_neigh = pointer.iloc[:, 48:pointer.shape[1]]
        file_cellid = pointer.iloc[:,1] # Cell ID
        file1preadjacency = pandas.concat([file_cellid, file_neigh], axis = 1) # Join cell IDs to neighbor data
        # Arrange into two-columns - cell ID and neighbor
        f12 = file1preadjacency.melt(id_vars = "CellId", value_vars = file1preadjacency.columns[1:], var_name = "NeighbourNumber", value_name = "NeighbourId")
        f13 = f12[f12.NeighbourId != 0].drop('NeighbourNumber', axis = 1) # Remove all non-neighbor lines
        
        relcols = pointer.columns[2:35] # we need columns 2:34
        vertex_tensor = torch.tensor(pointer.loc[:, relcols].values, dtype = torch.double)
        edge_tensor = torch.tensor(f13.transpose().values) #names = ("CellId", "NeighbourId"))
        dataset.append(Data(x = vertex_tensor, edge_index = edge_tensor, y = torch.tensor([int(label == "DCB")])))

# 2. Neighbor if within a fixed radius of center cell
# Entire slide is has coordinates ranging from x in (60, 700) to y in (30, 700)
radius = 25
for label in labels:
    folderpath = join(folder, label)
    for file in listdir(folderpath):
        pointer = pandas.read_csv(join(folderpath, file)) # Read in the data


computer = pointer[["CellId", "X_position", "Y_position"]]
computer.loc[0:,"neighbors"] = []
computer['neighbors'] = computer['neighbors'].astype(object) # So the column accepts list objects
for i in range(0, computer.shape[0]):
    x_i = computer.iloc[i, 1]
    y_i = computer.iloc[i, 2]
    nbd = (computer.X_position - x_i) ** 2 + (computer.Y_position - y_i) ** 2 < radius ** 2 # All the neighbors within radius of indexed cell
    nbd = list(computer.loc[[i for i, x in enumerate(nbd) if x], ["CellId"]].CellId) # Find indices of those neighbors
    computer.at[i, 'neighbors'] = nbd
# Now I have to convert it to a list of pairs. Or should i just convert to an adjacency matrix, since i'll be computing the laplacian and the fiedler valuek
from itertools import product
test = product(computer.neighbors[0], computer.neighbors[0])

neighborlist = []
for index, cell in computer.iterrows():
    neighborlist.append([[cell.CellId, i] for i in cell.neighbors])
neighborlist = [item for sublist in neighborlist for item in sublist]
neighborlist = [i for i in neighborlist if i[0] != i[1]]



# 3. Neighbor if edges of ellipses are within a certain distance of each other
distance = 5
computer = pointer[['CellId', 'X_position', 'Y_position', 'MajorAxisLength', 'MinorAxisLength', 'Orientation']]
computer.loc[:, 'Orientation'] = computer.Orientation / 180 * np.pi
# computer.loc[0:, 'neighbors'] = []
# computer['neighbors'] = computer['neighbors'].astype(object) # Prepare column to accept list objects

    
# Compute A B and C from standard form of ellipse
computer.loc[:, 'A'] = (np.cos(computer.Orientation) / (computer.MajorAxisLength / 2)) ** 2 + (np.sin(computer.Orientation) / (computer.MinorAxisLength / 2)) ** 2
computer.loc[:, 'B'] = 2 * np.cos(computer.Orientation) * np.sin(computer.Orientation) * ((2 / computer.MajorAxisLength) ** 2 - (2 / computer.MinorAxisLength) ** 2)
computer.loc[:, 'C'] = (np.sin(computer.Orientation) / (computer.MajorAxisLength / 2)) ** 2 + (np.cos(computer.Orientation) / (computer.MinorAxisLength / 2)) ** 2
for i in range(0, computer.shape[0]):
    x_i, y_i = computer.iloc[i,[1,2]]
    # Find solution to every other  ellipse
    computer.loc[:, 'slope'] = (computer.Y_position - y_i) / (computer.X_position - x_i)
    computer.loc[:, 'qa'] = computer.A + computer.B * computer.slope + computer.C * (computer.slope ** 2)
    computer.loc[:, 'qb'] = computer.slope * ( (computer.B + 2 * computer.C) * y_i - (computer.B + 2 * computer.C * computer.slope) * x_i)
    computer.loc[:, 'qc'] = computer.C * (x_i * computer.slope - y_i) ** 2 - 1
    computer.loc[:, 'qxplus'] = (-1 * computer.qb + np.sqrt(computer.qb ** 2 - 4 * computer.qa * computer.qc)) / (2 * computer.qa)
    computer.loc[:, 'qxminus'] = (-1 * computer.qb - np.sqrt(computer.qb ** 2 - 4 * computer.qa * computer.qc)) / (2 * computer.qa)
    # Find intersecting line with own ellipse (Use m from row of each cell with ABC of cell i)
    computer.loc[:, 'ia'] = computer.loc[i, 'A'] + computer.loc[i, 'B'] * computer.slope + computer.loc[i, 'C'] * (computer.slope ** 2)
    computer.loc[:, 'ib'] = computer.slope * ( (computer.loc[i, 'B'] + 2 * computer.loc[i, 'C']) * y_i - (computer.loc[i, 'B'] + 2 * computer.loc[i, 'C'] * computer.slope) * x_i)
    computer.loc[:, 'ic'] = computer.loc[i, 'C'] * (x_i * computer.slope - y_i) ** 2




# Distribution of major and minor axis lengths of cells
maj = pointer.MajorAxisLength
min = pointer.MinorAxisLength

fig, axs = plt.subplots(1, 2, sharey = True, tight_layout = True)
axs[0].hist(maj, bins = 10)
axs[1].hist(min, bins = 10)
axs[0].set_title("Major axes")
axs[1].set_title("Minor axes")

# Trying to get the graph laplacian and the Fiedler values
import datasets
from torch_geometric.utils import get_laplacian, to_networkx

dataset = datasets.c1Data()
data = dataset[0]
lap = get_laplacian(data.edge_index)

from torch_geometric.transforms import LaplacianLambdaMax

test = LaplacianLambdaMax()
more = test(data)

import networkx
import scipy

test = to_networkx(data, to_undirected=True)
lap = networkx.laplacian_matrix(test).astype('double')
scipy.sparse.linalg.eigs(lap)


fied = networkx.fiedler_vector(test)
bits = networkx.number_connected_components(test)
for data in dataset:
    ntkx = to_networkx(data, to_undirected = True)
    networkx.number_connected_components(ntkx)
