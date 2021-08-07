import pandas as pd
import numpy as np
import torch
import networkx as nx
from pandas.core.frame import DataFrame
from os import listdir
from os.path import isfile, join, splitext
from torch.utils.data import dataloader
from torch_geometric.data import Data, Dataset, DataLoader, batch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.nn.pool import radius
from torch_geometric.utils import to_networkx, from_networkx, subgraph
from itertools import product, compress

pd.options.mode.chained_assignment = None

# So i'm going to keep both the c1 and c2 datasets because they're convenient to be able to load. 
# Loads data from c1 dataset
class c1Data(InMemoryDataset):
    def __init__(self, root = "/data/kevin/IMC_Oct2020/", transform = None, pre_transform = None, pre_filter = None):
        super(c1Data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = "/data/kevin/IMC_Oct2020/c1/"
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
        folder = "/data/kevin/IMC_Oct2020/c1/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Data are split into responders (DCB) and nonresponders (NDB)
        for label in labels:
            folderpath = join(folder, label)
            for file in listdir(folderpath):
                pointer = pd.read_csv(join(folderpath, file)) # Read in the data
                # Construct adjacency matrix
                file_neigh = pointer.iloc[:, 48:pointer.shape[1]]
                file_cellid = pointer.iloc[:,1] # Cell ID
                file1preadjacency = pd.concat([file_cellid, file_neigh], axis = 1) # Join cell IDs to neighbor data
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
        folder = "/data/kevin/IMC_Oct2020/c2/"
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
        folder = "/data/kevin/IMC_Oct2020/c2/"
        labels = ['DCB', 'NDB']
        dataset = []

        # Data are split into responders (DCB) and nonresponders (NDB)
        for label in labels:
            folderpath = join(folder, label)
            for file in listdir(folderpath):
                pointer = pd.read_csv(join(folderpath, file)) # Read in the data
                # Construct adjacency matrix
                file_neigh = pointer.iloc[:, 48:pointer.shape[1]]
                file_cellid = pointer.iloc[:,1] # Cell ID
                file1preadjacency = pd.concat([file_cellid, file_neigh], axis = 1) # Join cell IDs to neighbor data
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

class neighbor_metric_class(InMemoryDataset):
    def __init__(self, root = "/data/kevin/IMC_Oct2020/", 
                    transform = None, pre_transform = None, pre_filter = None,
                    dataset = 'c2', neighbordef = 'initial',
                    naive_radius = 25,
                    knn = 5, knn_max = 50):
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

        super(neighbor_metric_class, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])                                

    @property
    def raw_file_names(self):
        folder = f"/data/kevin/IMC_Oct2020/{self.dataset}/"
        labels = ['DCB', 'NDB']
        names = listdir(join(folder, labels[0])) + listdir(join(folder, labels[1]))
        return names

    @property
    def processed_file_names(self):
        if self.neighbordef == 'naive':
            neighbor_metric = f'naive{self.naive_radius}'
        if self.neighbordef == 'knn':
            neighbor_metric = f'knn{self.knn}max{self.max_distance}'

        return [f'{self.dataset}_{neighbor_metric}.data']

    def download(self):
        # Leave this empty?
        return []

    def process(self):
        if self.dataset == 'c1':
            dataset = c1Data()
        elif self.dataset == 'c2':
            dataset = c2Data()

        new_neighbors = []
        if self.neighbordef == 'naive':
            for data in dataset:
                computer = DataFrame(data.pos.numpy(), columns=['X_position', 'Y_position']) # Read in the data
                computer.insert(0, 'CellId', [i for i in range(1, data.pos.shape[0] + 1)])
                computer.loc[0:,"neighbors"] = np.nan # Can ignore the value warning.
                computer['neighbors'] = computer['neighbors'].astype(object) # So the column accepts list objects
                for i in range(0, computer.shape[0]):
                    x_i = computer.iloc[i, 1]
                    y_i = computer.iloc[i, 2]
                    nbd = (computer.X_position - x_i) ** 2 + (computer.Y_position - y_i) ** 2 < self.naive_radius ** 2 # All the neighbors within radius of indexed cell
                    nbd = list(computer.loc[[i for i, x in enumerate(nbd) if x], ["CellId"]].CellId) # Find indices of those neighbors
                    computer.at[i, 'neighbors'] = nbd
                    
                # Now I have to convert it to a list of pairs. 
                neighborlist = []
                for index, cell in computer.iterrows():
                    neighborlist.append([[cell.CellId, i] for i in cell.neighbors])
                neighborlist = [item for sublist in neighborlist for item in sublist]
                neighborlist = np.array([i for i in neighborlist if i[0] != i[1]])
                edge_tensor = torch.tensor(neighborlist.transpose() - 1, dtype = torch.long) #names = ("CellId", "NeighbourId")) 

                new_neighbors.append(Data(x = data.x,
                                      y = data.y,
                                      edge_index = edge_tensor,
                                      pos = data.pos,
                                      name = data.name + '_naive' + str(self.naive_radius)))

        elif self.neighbordef == 'knn':
            for data in dataset:            
                computer = DataFrame(data.pos.np(), columns=['X_position', 'Y_position']) # Read in the data
                computer.insert(0, 'CellId', [i for i in range(1, data.pos.shape[0] + 1)])
                
                for index, row in computer.iterrows():
                    distance = np.sqrt((computer.X_position - row.X_position) ** 2 + (computer.Y_position - row.Y_position) ** 2)
                    sorts = sorted(enumerate(distance), key = lambda x:x[1])
                    targets = [sorts [i][0] for i in range(1, self.knn + 1)]
                    edges = edges + [[index, i] for i in targets] + [[i, index] for i in targets]
                # Find unique edges
                unique_edges = [list(x) for x in set(tuple(x) for x in edges)]
                edge_tensor = torch.tensor(np.array(unique_edges).transpose(), dtype = torch.long)

                new_neighbors.append(Data(x = data.x,
                                    y = data.y,
                                    edge_index = edge_tensor,
                                    pos = data.pos,
                                    name = data.name + '_naive' + str(25)))
                                    
        
        data, slices = self.collate(new_neighbors)
        torch.save((data, slices), self.processed_paths[0])



# Function that finds all neighbors a distance 'n' from node 'node' on graph 'G' 
def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items()
                    if length == n]

class IMC_Data(InMemoryDataset):
    def __init__(self, root = "/data/kevin/IMC_Oct2020/", 
                 transform = None, pre_transform = None, pre_filter = None,
                 dataset = 'c2', neighbordef = 'initial', subgraph = 'windows',
                 naive_radius = 25,
                 knn = 5, knn_max = 50,
                 high_exp_marker = 'CK', exp_radius = 5, exp_depth = 10,
                 width = 250, window_num = 10, min_cells = 50):
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
            self.min_cells = min_cells
        elif subgraph != 'none':
            raise print('Pick a valid subgraph method.')

        super(IMC_Data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        folder = f"/data/kevin/IMC_Oct2020/{self.dataset}/"
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
        if self.neighbordef == 'initial':
            if self.dataset == 'c1':
                dataset = c1Data()
            elif self.dataset == 'c2':
                dataset = c2Data()
        elif self.neighbordef == 'naive':
            dataset = neighbor_metric_class(dataset = self.dataset, neighbordef = self.neighbordef,
                                  naive_radius = self.naive_radius)
        elif self.neighbordef == 'knn':
            dataset = neighbor_metric_class(dataset = self.dataset, neighbordef = self.neighbordef,
                                  knn = self.knn, knn_max = self.max_distance)
            
        
        if self.subgraph == 'high_exp':
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
                    unique_nodes = np.unique(subgraph_edges[1,:].tolist()).tolist()
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
        elif self.subgraph == 'windows':
            subgraph_dataset = []
            data_counter = 1
            for data in dataset:
                sample_dims = [data.pos[:,0].min(), data.pos[:,0].max() - self.window_width,
                               data.pos[:,1].min(), data.pos[:,1].max() - self.window_width] 
                               # x_min, x_max (adj for window width), y_min, y_max (adj for window width)
                counter = 1
                window_counter = 1
                while counter < self.window_number + 1:
                    if window_counter > 1000:
                        print(f'Exceeded 1000 attempts on graph {data_counter}, skipping.')
                        break
                    window_counter += 1
                    window_x = np.random.random() * (sample_dims[1] - sample_dims[0]) + sample_dims[0]
                    window_y = np.random.random() * (sample_dims[3] - sample_dims[2])+ sample_dims[2]
                    window_dims = [window_x, window_x + self.window_width,
                                   window_y, window_y + self.window_width]
                    
                    x_inc = (data.pos[:,0] > window_dims[0]) & (data.pos[:,0] < window_dims[1])
                    y_inc = (data.pos[:,1] > window_dims[2]) & (data.pos[:,1] < window_dims[3])
                    mask = x_inc & y_inc
                    if sum(mask) < self.min_cells:
                        continue
                    
                    # Subset
                    nodes = data.x[mask,:]
                    pos = data.pos[mask,:]
                    node_indices = [i for i,x in enumerate(mask) if x]
                    ndata = to_networkx(data)
                    nsub = ndata.subgraph(node_indices)
                    edges = np.array([[i,x] for i,x in nsub.edges]).transpose()
                    d = dict((x,i) for i,x in enumerate(node_indices))
                    for q in range(0, edges.shape[1]):
                        edges[0,q] = d[edges[0,q]]
                        edges[1,q] = d[edges[1,q]]
                    subgraph_dataset.append(Data(x = nodes,
                                                 y = data.y,
                                                 edge_index = torch.tensor(edges, dtype = torch.long),
                                                 pos = pos,
                                                 name = data.name + '_window' + str(counter)))
                    counter += 1
                
                print(f'Windows for graph {data_counter} of {len(dataset)} completed after {window_counter} windows attempted.')
                data_counter += 1
     
        data, slices = self.collate(subgraph_dataset)
        torch.save((data, slices), self.processed_paths[0])
