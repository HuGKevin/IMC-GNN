import pandas, numpy, torch
from pandas.core.frame import DataFrame
from os import listdir
from os.path import isfile, join, splitext
from torch.utils.data import dataloader
from torch_geometric.data import Data, Dataset, DataLoader, batch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, subgraph
import networkx as nx

# We will train the same GNN model on subgraphs of the original dataset
# Subgraphs will be created by selecting the cell with the highest cytokeratin (PANCK) expression

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
                dataset.append(Data(x = vertex_tensor, 
                                    edge_index = edge_tensor, 
                                    y = torch.tensor([int(label == "DCB")]), 
                                    name = splitext(file)[0]))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

# Dataset of subgraphs selected by taking 
class c1_ck_subgraphs(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c1_ck_subgraphs, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        return ['c1_ck_subgraphs.data']

    def download(self):
        # Leave this empty?
        return []

    def neighborhood(G, node, n):
        path_lengths = nx.single_source_dijkstra_path_length(G, node)
        return [node for node, length in path_lengths.items()
                    if length == n]

    def process(self):
        dataset = c1Data()
        nbd_depth = 5
        subgraph_dataset = []

        # Set up for finding subgraphs defined by neighborhood breadth
        for k in range(0, len(dataset)):
            pt_data = dataset[k]
            panck_ranks = DataFrame(data = {'panck': pt_data.x[:,6].tolist(), 'index':[i for i in range(0,pt_data.x.shape[0])]}) # col 6 is that of PANCK
            nx_data = to_networkx(dataset[k], to_undirected= True)
            results = DataFrame(columns = ['Max_node', 'Max_PANCK', 'Neighborhood_nodes'])

            # Identify the nodes in each one
            for j in range(0, 10):
                max_node = panck_ranks.index[panck_ranks.panck.argmax()]
                max_panck = panck_ranks.panck[max_node]
                subgraphlist = [max_node]
                for i in range(0, nbd_depth):
                    subgraphlist.extend(neighborhood(nx_data, max_node, i + 1))
                panck_ranks = panck_ranks.drop(subgraphlist, errors = 'ignore')
                # panck_ranks.drop([i for i,x in enumerate(panck_ranks.index) if x in subgraphlist])
                addition = {'Max_node':max_node, 'Max_PANCK':max_panck, 'Neighborhood_nodes':subgraphlist}
                results = results.append(addition, ignore_index = True)

            # Extract the edge and node features for each subgraph
            for p in range(0, results.shape[0]):
                subgraph_edges = subgraph(results.Neighborhood_nodes[p], pt_data.edge_index)[0]
                unique_nodes = numpy.unique(subgraph_edges[1,:].tolist()).tolist()
                for i,x in enumerate(unique_nodes):
                    print(i, x)
                    for q in range(0, subgraph_edges.shape[1]):
                        if subgraph_edges[0,q] == x:
                            subgraph_edges[0,q] = i
                        if subgraph_edges[1,q] == x:
                            subgraph_edges[1,q] = i
                subgraph_nodes = pt_data.x[results.Neighborhood_nodes[p],:]
                subgraph_dataset.append(Data(x = subgraph_nodes,
                                            y = pt_data.y,
                                            edge_index = subgraph_edges,
                                            name = pt_data.name + '_Sub' + str(p)))


        data, slices = self.collate(subgraph_dataset)
        torch.save((data, slices), self.processed_paths[0])

# Function that finds all neighbors a distance 'n' from node 'node' on graph 'G' 
def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
                    if length == n]

# Implement a GNN on the above set of subgraphs to see how well it can predict response to treatment

# Necessary packages
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from random import randint

dataset = c1_ck_subgraphs()

# Define test and train sets - note that we're splitting up by patients, so that they stay in blocks of ten
torch.manual_seed(12345)
test_index = []
for j in [j for j in [randint(0, len(dataset)/10) for i in range(0, 4)]]:
    test_index.extend([i for i in range(10 * j, 10 * j+10)])
train_index = [i for i in range(0, len(dataset)) if i not in test_index]

test_dataset = dataset[test_index]
train_dataset = dataset[train_index]

print(f"Number of test graphs: {len(test_dataset)}")
print(f"Number of training graphs: {len(train_dataset)}")

# Batching
train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 10, shuffle = False)

# Build the GNN (Using the same architecture as from our first functional GNN)
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

model = GCN(hidden_channels=16).double()
print(model)

# Training the model
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

tracker = []
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
