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
from itertools import product

pandas.options.mode.chained_assignment = None

# So i'm going to keep both the c1 and c2 datasets because they're convenient to be able to load. 
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
                pos_tensor = torch.tensor(pointer.loc[:, ['X_position', 'Y_position']].values, dtype = torch.double)
                dataset.append(Data(x = vertex_tensor, 
                                    edge_index = edge_tensor, 
                                    y = torch.tensor([int(label == "DCB")]), 
                                    pos = pos_tensor,
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
                pos_tensor = torch.tensor(pointer.loc[:, ['X_position', 'Y_position']].values, dtype = torch.double)
                dataset.append(Data(x = vertex_tensor, 
                                    edge_index = edge_tensor, 
                                    y = torch.tensor([int(label == "DCB")]),
                                    pos = pos_tensor,
                                    name = splitext(file)[0]))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

# Constructs subgraphs of c1 dataset based on highest CK expression.
# Takes arguments radius = width of neighborhood; depth = how many neighborhoods to take. Note: can't seem to properly implement these arguments.
# Bug fixed - turns out putting the instance variable definition before the inheritance call (super().__init__()) makes a difference. Not sure why.
class c1_ck_subgraphs(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None, radius = 5, depth = 10):
        self.radius = radius 
        self.depth = depth 
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
        name = f"c1_ck_r{self.radius}_d{self.depth}_subgraphs.data"
        return [name]

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
            for j in range(0, self.depth):
                max_node = panck_ranks.index[panck_ranks.panck.argmax()]
                max_panck = panck_ranks.panck[max_node]
                subgraphlist = [max_node]
                for i in range(0, self.radius):
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
                subgraph_pos = pt_data.pos[results.Neighborhood_nodes[p],:]
                subgraph_nodes = pt_data.x[results.Neighborhood_nodes[p],:]
                subgraph_dataset.append(Data(x = subgraph_nodes,
                                            y = pt_data.y,
                                            edge_index = subgraph_edges,
                                            pos = subgraph_pos,
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
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None,):
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
                    pos_tensor = torch.tensor(pointer.iloc[window_cellid - 1,].loc[:, ['X_position', 'Y_position']].values, dtype = torch.double)
                    dataset.append(Data(x = node_tensor, 
                                        edge_index = edge_tensor,
                                        y = torch.tensor([int(label == "DCB")]),
                                        pos = pos_tensor,
                                        name = splitext(file)[0] + f"_area{p}"))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])


class c1_naive_neighbors(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None,radius = 25):
        self.radius = radius
        super(c1_naive_neighbors, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        return [f'c1_naive_neighbors_r{self.radius}.data']

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
                computer = pointer[["CellId", "X_position", "Y_position"]]
                computer.loc[0:,"neighbors"] = numpy.nan # Can ignore the value warning.
                computer['neighbors'] = computer['neighbors'].astype(object) # So the column accepts list objects
                for i in range(0, computer.shape[0]):
                    x_i = computer.iloc[i, 1]
                    y_i = computer.iloc[i, 2]
                    nbd = (computer.X_position - x_i) ** 2 + (computer.Y_position - y_i) ** 2 < self.radius ** 2 # All the neighbors within radius of indexed cell
                    nbd = list(computer.loc[[i for i, x in enumerate(nbd) if x], ["CellId"]].CellId) # Find indices of those neighbors
                    computer.at[i, 'neighbors'] = nbd
                    
                # Now I have to convert it to a list of pairs. 
                neighborlist = []
                for index, cell in computer.iterrows():
                    neighborlist.append([[cell.CellId, i] for i in cell.neighbors])
                neighborlist = [item for sublist in neighborlist for item in sublist]
                neighborlist = numpy.array([i for i in neighborlist if i[0] != i[1]])
                
                relcols = pointer.columns[2:35] # we need columns 2:34
                vertex_tensor = torch.tensor(pointer.loc[:, relcols].values, dtype = torch.double)
                edge_tensor = torch.tensor(neighborlist.transpose() - 1, dtype = torch.long) #names = ("CellId", "NeighbourId")) 
                pos_tensor = torch.tensor(pointer.loc[:, ['X_position', 'Y_position']].values, dtype = torch.double)
                dataset.append(Data(x = vertex_tensor, 
                                    edge_index = edge_tensor, 
                                    y = torch.tensor([int(label == "DCB")]), 
                                    pos = pos_tensor,
                                    name = splitext(file)[0]))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

class c1_naive_ck_subgraphs(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None, radius = 5, depth = 10, naive_radius = 25):
        self.radius = radius 
        self.depth = depth 
        self.naive = naive_radius
        super(c1_naive_ck_subgraphs, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        name = f"c1_naive_ck_r{self.radius}_d{self.depth}_subgraphs.data"
        return [name]

    def download(self):
        # Leave this empty?
        return []

    def neighborhood(G, node, n):
        path_lengths = nx.single_source_dijkstra_path_length(G, node)
        return [node for node, length in path_lengths.items()
                    if length == n]

    def process(self):
        dataset = c1_naive_neighbors(radius = self.naive)
        subgraph_dataset = []

        # Set up for finding subgraphs defined by neighborhood breadth
        for k in range(0, len(dataset)):
            pt_data = dataset[k]
            panck_ranks = DataFrame(data = {'panck': pt_data.x[:,6].tolist(), 'index':[i for i in range(0,pt_data.x.shape[0])]}) # col 6 is that of PANCK
            nx_data = to_networkx(dataset[k], to_undirected= True)
            results = DataFrame(columns = ['Max_node', 'Max_PANCK', 'Neighborhood_nodes'])

            # Identify the nodes in each one
            for j in range(0, self.depth):
                max_node = panck_ranks.index[panck_ranks.panck.argmax()]
                max_panck = panck_ranks.panck[max_node]
                subgraphlist = [max_node]
                for i in range(0, self.radius):
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
                subgraph_pos = pt_data.pos[results.Neighborhood_nodes[p],:]
                subgraph_nodes = pt_data.x[results.Neighborhood_nodes[p],:]
                subgraph_dataset.append(Data(x = subgraph_nodes,
                                            y = pt_data.y,
                                            edge_index = subgraph_edges,
                                            pos = subgraph_pos,
                                            name = pt_data.name + '_Sub' + str(p)))

            print(f"Graph {k} of {len(dataset)} processed")

        data, slices = self.collate(subgraph_dataset)
        torch.save((data, slices), self.processed_paths[0])


class IMC_Data(InMemoryDataset):
    def __init__(self, root = "C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/", 
                 transform = None, pre_transform = None, pre_filter = None,
                 dataset = 'c2', neighbordef = 'initial', subgraph = 'windows',
                 naive_radius = 25,
                 knn = 5, knn_max = 50,
                 high_exp_marker = 'CK', exp_radius = 5, exp_depth = 10,
                 width = 250, window_num = 10):
        self.dataset = dataset
        
        self.neighbordef = neighbordef
        if self.neighbordef == 'naive':
            self.naive_radius = naive_radius
        elif self.neighbordef == 'knn':
            self.knn = knn
            self.max_distance = knn_max
        elif self.neighbordef == 'ellipse':
            print("Coming soon to a theater near you.")
        else:
            raise print("Choose a valid neighbor metric.")
        
        self.subgraph = subgraph
        if subgraph == 'high_exp':
            self.marker = high_exp_marker
            self.radius = exp_radius
            self.depth = exp_depth
        elif subgraph == 'windows':
            self.window_width = width
            self.window_number = window_num
        elif subgraph != 'none':
            raise print('Pick a valid subgraph method.')

        super(IMC_Data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = f"C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/{self.dataset}/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        if self.neighbordef == 'naive':
            neighbor_metric = f'naive{self.naive_radius}'
        if self.neighbordef == 'knn':
            neighbor_metric = f'knn{self.knn}max{self.max_distance}'

        if self.subgraph == 'high_exp':
            subgraph_method = f'{self.marker}r{self.radius}d{self.depth}'
        elif self.subgraph == 'windows':
            subgraph_method = f'width{self.window_width}n{self.window_number}'

        return [f'{self.dataset}_{neighbor_metric}_{subgraph_method}.data']

    def download(self):
        # Leave this empty?
        return []

    def process(self):
        if self.dataset == 'c1':
            dataset = c1Data()
        elif self.dataset == 'c2':
            dataset = c2Data()
    
        if self.neighbordef == 'naive':
            for data in dataset:
                computer = DataFrame(data.pos.numpy(), columns=['X_position', 'Y_position']) # Read in the data
                computer.insert(0, 'CellId', [i for i in range(1, data.pos.shape[0] + 1)])
                computer.loc[0:,"neighbors"] = numpy.nan # Can ignore the value warning.
                computer['neighbors'] = computer['neighbors'].astype(object) # So the column accepts list objects
                for i in range(0, computer.shape[0]):
                    x_i = computer.iloc[i, 1]
                    y_i = computer.iloc[i, 2]
                    nbd = (computer.X_position - x_i) ** 2 + (computer.Y_position - y_i) ** 2 < self.radius ** 2 # All the neighbors within radius of indexed cell
                    nbd = list(computer.loc[[i for i, x in enumerate(nbd) if x], ["CellId"]].CellId) # Find indices of those neighbors
                    computer.at[i, 'neighbors'] = nbd
                    
                # Now I have to convert it to a list of pairs. 
                neighborlist = []
                for index, cell in computer.iterrows():
                    neighborlist.append([[cell.CellId, i] for i in cell.neighbors])
                neighborlist = [item for sublist in neighborlist for item in sublist]
                neighborlist = numpy.array([i for i in neighborlist if i[0] != i[1]])
                edge_tensor = torch.tensor(neighborlist.transpose() - 1, dtype = torch.long) #names = ("CellId", "NeighbourId")) 
        elif self.neighbordef == 'knn':
            
        
        if self.subgraph == 'high_exp':

        elif self.subgraph == 'windows':



        
        folder = f"C:/Users/Kevin Hu/Desktop/Kluger/data/IMC_Oct2020/{self.dataset}/"
        labels = ['DCB', 'NDB']
        temp_dataset = []

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
                pos_tensor = torch.tensor(pointer.loc[:, ['X_position', 'Y_position']].values, dtype = torch.double)
                temp_dataset.append(Data(x = vertex_tensor, 
                                    edge_index = edge_tensor, 
                                    y = torch.tensor([int(label == "DCB")]), 
                                    pos = pos_tensor,
                                    name = splitext(file)[0]))

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
