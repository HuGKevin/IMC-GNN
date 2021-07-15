import pandas
import numpy
import torch
from os import listdir
from os.path import isfile, join
from torch.utils.data import dataloader
from torch_geometric.data import Data, Dataset, DataLoader, batch
from torch_geometric.data.in_memory_dataset import InMemoryDataset

class c1Data(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c1Data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        return ['c1.data']

    def download(self):
        # Leave this empty?
        return []

    def process(self):
        # Move all the for loops in here.
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Working just through the data in c1/DCB
        # Note: we have two chunks of data (c1, c2) which are split into responders (DCB) and nonresponders (NDB)
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
                edge_tensor = torch.tensor(f13.transpose().values - 1) #names = ("CellId", "NeighbourId")) 
                dataset.append(Data(x = vertex_tensor, edge_index = edge_tensor, y = torch.tensor([int(label == "DCB")])))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

# Instantiate the dataset
dataset = c1Data()

# Define training and test sets
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:24]
test_dataset = dataset[24:]

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

# Batching
train_loader = DataLoader(train_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

for step, data in enumerate(train_loader):
    print(f"Step {step + 1}")
    print("========")
    print(f"Number of graphs in the current batch: {data.num_graphs}")
    print(data)
    print()

# Build the GNN
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch) #[batch_size, hidden_channels]

        # 3. Final classifier
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x)

        return x

model = GCN(hidden_channels = 16).double()
print(model)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim = 1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)






# What happens if we apply that GNN to the other dataset?
# Load c2 data.
class c2Data(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c2Data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c2/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        return ['c2.data']

    def download(self):
        # Leave this empty?
        return []

    def process(self):
        # Move all the for loops in here.
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c2/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Working just through the data in c1/DCB
        # Note: we have two chunks of data (c1, c2) which are split into responders (DCB) and nonresponders (NDB)
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
                edge_tensor = torch.tensor(f13.transpose().values - 1) #names = ("CellId", "NeighbourId")) 
                dataset.append(Data(x = vertex_tensor, edge_index = edge_tensor, y = torch.tensor([int(label == "DCB")])))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])


c2 = c2Data()
c2loader = DataLoader(c2, batch_size=5, shuffle = False)
gen_acc = test(c2loader)


# What if we train on c2, since we have more samples in it? Give us a bigger test set
dataset = c2Data()
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:31]
test_dataset = dataset[31:]

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

# Batching
train_loader = DataLoader(train_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

model = GCN(hidden_channels = 16).double()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()
tracker = []

for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

tracker = pandas.DataFrame(tracker)
tracker.columns = ['epoch', 'train', 'test']

import matplotlib.pyplot as plt