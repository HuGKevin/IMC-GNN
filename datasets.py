import pandas, numpy, torch
from pandas.core.frame import DataFrame
from os import listdir
from os.path import isfile, join, splitext
from torch.utils.data import dataloader
from torch_geometric.data import Data, Dataset, DataLoader, batch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.nn.pool import radius
from torch_geometric.utils import to_networkx, from_networkx, subgraph
import networkx as nx

pandas.options.mode.chained_assignment = None

# Loads data from c1 dataset
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
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Data are split into responders (DCB) and nonresponders (NDB)
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

# Loads data from c2 dataset
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
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c2/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Data are split into responders (DCB) and nonresponders (NDB)
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

# Constructs subgraphs of c1 dataset based on highest CK expression.
# Takes arguments radius = width of neighborhood; depth = how many neighborhoods to take. Note: can't seem to properly implement these arguments.
class c1_ck_subgraphs(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c1_ck_subgraphs, self).__init__(root, transform, pre_transform, pre_filter)
        # self.radius = radius
        # self.depth = depth
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        # name = f"c1_ck_r{self.radius}_d{self.depth}_subgraphs.data"
        return ["c1_ck_subgraphs.data"]

    def download(self):
        # Leave this empty?
        return []

    def neighborhood(G, node, n):
        path_lengths = nx.single_source_dijkstra_path_length(G, node)
        return [node for node, length in path_lengths.items()
                    if length == n]

    def process(self):
        dataset = c1Data()
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
                for i in range(0, 5):
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

            print(f"Graph {k} of {len(dataset)} processed")

        data, slices = self.collate(subgraph_dataset)
        torch.save((data, slices), self.processed_paths[0])

# Function that finds all neighbors a distance 'n' from node 'node' on graph 'G' 
def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
                    if length == n]

class c1_area_subgraphs(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c1_area_subgraphs, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        return ['c1_area_subgraphs.data']

    def download(self):
        # Leave this empty?
        return []

    def process(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Data are split into responders (DCB) and nonresponders (NDB)
        for label in labels:
            folderpath = join(folder, label)
            for file in listdir(folderpath):
                pointer = pandas.read_csv(join(folderpath, file)) # Read in the data
                xmin = pointer.X_position.min()
                ymin = pointer.Y_position.min()
                x_width = (pointer.X_position.max() - pointer.X_position.min()) / 5
                y_width = (pointer.Y_position.max() - pointer.Y_position.min()) / 5
                x = [xmin + i * x_width for i in range(0, 6)]
                y = [ymin + i * y_width for i in range(0, 6)]
                # Thirteen windows in a pyramid shape around the center one.
                # Each item is a window in the format [x_min, x_max, y_min, y_max]
                windows = [[x[2], x[3], y[0], y[1]],
                           [x[1], x[2], y[1], y[2]],
                           [x[2], x[3], y[1], y[2]],
                           [x[3], x[4], y[1], y[2]],
                           [x[0], x[1], y[2], y[3]],
                           [x[1], x[2], y[2], y[3]], 
                           [x[2], x[3], y[2], y[3]],
                           [x[3], x[4], y[2], y[3]],
                           [x[4], x[5], y[2], y[3]],
                           [x[1], x[2], y[3], y[4]],
                           [x[2], x[3], y[3], y[4]],
                           [x[3], x[4], y[3], y[4]],
                           [x[2], x[3], y[4], y[5]]]

                for p in range(0, len(windows)):
                    window = windows[p]
                    pointer_window = pointer[(pointer.X_position > window[0]) & 
                                             (pointer.X_position < window[1]) & 
                                             (pointer.Y_position > window[2]) & 
                                             (pointer.Y_position < window[3])]

                    window_neigh = pointer_window.iloc[:, 48:pointer_window.shape[1]]
                    window_cellid = pointer_window.iloc[:,1]
                    window_preadj = pandas.concat([window_cellid, window_neigh], axis = 1)
                    w12 = window_preadj.melt(id_vars = "CellId", value_vars = window_preadj.columns[1:], var_name = "NeighbourNumber", value_name = "NeighbourId") # Convert to two-column edges
                    w13 = w12[w12.NeighbourId != 0].drop('NeighbourNumber', axis = 1) # Drop the empty edges
                    w14 = w13[w13.NeighbourId.isin(w13.CellId)] # Drop edges that go to nodes not in the window
                    # Reindex the edges such that they match node dimensions
                    unique_nodes = numpy.unique(w14.CellId.tolist()) 
                    for i,x in enumerate(unique_nodes):
                        for q in range(0, w14.shape[0]): # Need to figure out how to suppress warnings here, or at least understand the warning.
                            if w14.iloc[q,0] == x:
                                w14.iloc[q,0] = i
                            if w14.iloc[q,1] == x:
                                w14.iloc[q,1] = i

                    colnames = pointer_window.columns[2:35]
                    node_tensor = torch.tensor(pointer_window.loc[:, colnames].values, dtype = torch.double)
                    edge_tensor = torch.tensor(w14.transpose().values)
                    dataset.append(Data(x = node_tensor, 
                                        edge_index = edge_tensor,
                                        y = torch.tensor([int(label == "DCB")]),
                                        name = splitext(file)[0] + f"_area{p}"))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
