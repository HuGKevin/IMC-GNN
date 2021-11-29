import sys
sys.path.append('/home/kevin/project')

from itertools import count
import torch
from IMC_GNN import datasets
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch.nn import Linear
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np

# Class for GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels) ### Things like dataset features can be passed through as arguments. Will probably simplify things. 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)

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

# Class for GCN model
class test_GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels) ### Things like dataset features can be passed through as arguments. Will probably simplify things. 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = SAGPooling(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.pool(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch) #[batch_size, hidden_channels]

        # 3. Final classifier
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x)

        return x


# Class for model with customizable architecture
class new_GCN(torch.nn.Module):
    def __init__(self, arch_dict, output_dim):
        # arch_dict is a list of 2-ples of form [layer, input_channels]
        super().__init__()
        self.arch_dict = arch_dict
        self.architecture = []

        
        
        for index in range(0, len(arch_dict)):
            if index == len(arch_dict) - 1:
                self.architecture.append(arch_dict[index][0](arch_dict[index][1], output_dim))
            else:
                self.architecture.append(arch_dict[index][0](arch_dict[index][1], arch_dict[index + 1][1]))
        
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

# Class for trainer class
class GCN_Train():
    def __init__(self, 
                 input_dim, output_dim, hidden_channels, 
                 metric = 'auc', lr = 0.001, loss_fxn = 'cel', optimizer = 'adam',
                 es_thresh = 0.03, es_lambda = 0.8, es_min_iter = 100, es_stall_limit = 5):
        self.model = GCN(input_dim = input_dim, output_dim = output_dim, hidden_channels = hidden_channels).double()
        self.device = torch.device('cpu')

        self.metric = metric
        self.flagraiser = False
        self.train_acc = [] # Note that validation accuracy is tracked in the flag raiser object

        if loss_fxn == 'cel':
            self.criterion = torch.nn.CrossEntropyLoss()
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        self.flag = EarlyStopFlag(es_thresh, es_lambda, es_min_iter, es_stall_limit, metric = self.metric)

    def to(self, dev, verbose = False):
        self.device = torch.device(dev)
        self.model = self.model.to(self.device)
        if verbose:
            print(f"Model moved to {dev}")

    # Trains model for one epoch
    def train_epoch(self, train_DL, verbose = False, ignore_es = False):
        if self.flagraiser == True and ignore_es == False:
            print("Early stopping triggered - are you sure you want to continue?")
            return False

        self.model.train() # Puts model in training mode

        batch = 1
        for data in train_DL:
            if self.device != torch.device('cpu'):
                data = data.to(self.device)
            self.out = self.model(data.x, data.edge_index, data.batch) # torch.nn.Module is callable, and is defined to invoke forward(). 
            loss = self.criterion(self.out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if verbose == True:
                print(f"Batch #{batch} complete")
                batch += 1

        # Find training accuracy
        self.model.eval()
        pred = []
        true = []

        for data in train_DL:
            if self.device != torch.device('cpu'):
                data = data.to(self.device)
            pred.append(self.predict(data.x, data.edge_index, data.batch))
            true.append(data.y)

        final_pred = torch.cat(pred, dim = 0)
        final_true = torch.cat(true, dim = 0)

        self.train_acc.append(roc_auc_score(final_true.to(torch.device('cpu')),
                                            final_pred.to(torch.device('cpu'))))

    def validate(self, valid_DL, verbose = False):
        self.model.eval()

        pred = []
        true = []

        for data in valid_DL:
            if self.device != torch.device('cpu'):
                if verbose:
                    print(f'moving data to {self.device}')
                data = data.to(self.device)
            pred.append(self.predict(data.x, data.edge_index, data.batch))
            true.append(data.y)

        if verbose == True:
            print('---Predictions complete, computing metrics')

        final_pred = torch.cat(pred, dim = 0)
        final_true = torch.cat(true, dim = 0)

        if self.metric == 'auc':
            if verbose == True:
                print('---Computing ROC AUC')
                
            score = roc_auc_score(final_true.to(torch.device('cpu')),
                                  final_pred.to(torch.device('cpu')))
            self.flagraiser = self.flag.update(score, verbose = verbose)

        if self.metric == 'accuracy':
            if verbose == True:
                print('---Computing accuracy')
                
            score = (sum(torch.eq(final_pred, final_true)) / final_pred.shape[0]).item()
            self.flagraiser = self.flag.update(score, verbose = verbose)

        if verbose == True:
            print('---Validation complete')
            
        return score
            
    # Predicts classification based on current model parameters
    def predict(self, x, edge_index, batch):
        self.model.eval()
        out = self.model(x, edge_index, batch)
        m = torch.nn.Softmax(dim = 1)
        pred = m(out)[:,1]
        return pred

    # Same as valid() but doesn't update any metric trackers.
    def test(self, valid_DL, verbose = False):
        self.model.eval()

        pred = []
        true = []

        for data in valid_DL:
            if self.device != torch.device('cpu'):
                data = data.to(self.device)
            pred.append(self.predict(data.x, data.edge_index, data.batch))
            true.append(data.y)

        final_pred = torch.cat(pred, dim = 0)
        final_true = torch.cat(true, dim = 0)

        if self.metric == 'auc':
            score = roc_auc_score(final_true.to(torch.device('cpu')),
                                  final_pred.to(torch.device('cpu')))

        return score

    def load_model(self, path_to_model):
        self.model.load_state_dict(torch.load(path_to_model))
        self.model.eval()
        
    def save_model(self, path_to_model):
        torch.save(self.model.state_dict(), path_to_model)


class EarlyStopFlag():
    def __init__(self, threshold, lam, minimum_iters, stall_lim, metric = 'auc', max_mode = False, name = None):
        self.auc_track = []
        self.ema_track = []
        self.count = 0

        self.__metric__ = metric
        self.__lam__ = lam
        self.__thresh__ = threshold
        self.__minimum_iters__ = minimum_iters
        self.__max_mode__ = max_mode
        self.__stall_lim__ = stall_lim
        self.__name__ = name if name is not None else "Metric." + str(np.random.randint(10000))

    def update(self, new_value, verbose = False):
        # Return True if early stopping is triggered, False if not yet triggered
        if verbose == True:
            print("========Early Stopping========")

        self.auc_track.append(new_value)
        
        if verbose == True:
            print(f"{self.__metric__}: {new_value}")
            print(f"Epoch number {self.count + 1}")

        if self.count == 0:
            new_ema = new_value
            if verbose == True:
                print("Added first value")
        else:   
            new_ema = self.auc_track[-1] * self.__lam__ + self.ema_track[-1] * (1 - self.__lam__)
            if verbose == True:
               print(f"New EMA computed: {new_ema}")
            threshold = max(self.ema_track) * (1 - self.__thresh__)
            if verbose == True:
                print(f"New ema is {new_ema}, while threshold is {threshold}")

        self.count += 1
        self.ema_track.append(new_ema)

        if self.count < self.__minimum_iters__:
            if verbose == True:
                print("Still under minimum iterations threshold")
            return False

        if self.stall_test(self.auc_track, self.__stall_lim__):
            print(f"AUC has not changed over past {self.__stall_lim__} iterations, early stopping triggered.")
            return True
        
        if new_ema < threshold:
            return True
        else:
            return False

    def stall_test(self, auc_list, stall_lim):
        return(len(set(auc_list[-stall_lim:]))) == 1

    def reset(self):
        self.auc_track = []
        self.ema_track = []
        self.count = 0
