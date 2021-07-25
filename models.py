from itertools import count
import torch
import datasets
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np

# Class for GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels):
        super(GCN, self).__init__()
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

# Class for trainer class
class GCN_Train():
    def __init__(self, input_dim, output_dim, hidden_channels, metrics = 'auc', lr = 0.001, loss_fxn = 'cel', optimizer = 'adam'):
        self.model = GCN(input_dim = input_dim, output_dim = output_dim, hidden_channels = hidden_channels).double()
        self.metric = metrics
        self.flagraiser = False

        if loss_fxn == 'cel':
            self.criterion = torch.nn.CrossEntropyLoss()
        
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        self.flag = EarlyStopFlag(0.03, 0.8, 30)

    # Trains model for one epoch
    def train(self, train_DL, verbose = True):
        self.model.train() # Puts model in training mode

        batch = 1
        for data in train_DL:
            self.out = self.model(data.x, data.edge_index, data.batch) # torch.nn.Module is callable, and is defined to invoke forward(). 
            print(self.out)
            print(data.y)
            loss = self.criterion(self.out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if verbose == True:
                print(f"Batch #{batch} complete")
                batch += 1

        pred = []
        true = []

        for data in train_DL:
            pred.append(self.predict(data.x, data.edge_index, data.batch))
            true.append(data.y)

        final_pred = torch.cat(pred, dim = 0)
        final_true = torch.cat(true, dim = 0)

        self.flagraiser = self.flag.update(roc_auc_score(final_true, final_pred))

    def fit(self, x, y):
        return True
    
    def validate(self, valid_loader):
        self.model.eval()

        pred = []
        true = []

        for data in valid_loader:
            pred.append(self.predict(data.x, data.edge_index, data.batch))
            true.append(data.y)

        final_pred = torch.cat(pred, dim = 0)
        final_true = torch.cat(true, dim = 0)

        if self.metric == 'auc':
            score = roc_auc_score(final_true, final_pred)

        return score
            
    # Predicts classification based on current model parameters
    def predict(self, x, edge_index, batch):
        out = self.model(x, edge_index, batch)
        pred = out.argmax(dim = 1)
        return pred


class EarlyStopFlag():
    def __init__(self, threshold, lam, minimum_iters, max_mode = False, name = None):
        self.auc_track = []
        self.ema_track = []
        self.count = 0
        
        self.__lam__ = lam
        self.__thresh__ = threshold
        self.__minimum_iters__ = minimum_iters
        self.__max_mode__=max_mode
        self.__name__ = name if name is not None else "Metric." + str(np.random.randint(10000))

    def update(self, new_value):
        print(f"=================Input value is {new_value}===============")
        self.auc_track.append(new_value)
        print(f"self.count = {self.count}")

        if self.count == 0:
            new_ema = new_value
            print("Added first value")
        else:   
            new_ema = self.auc_track[-1] * self.__lam__ / (1 + self.count) + self.ema_track[-1] * (1 - self.__lam__ / (1 + self.count))
            print(f"New EMA computed: {new_ema}")
            threshold = max(self.ema_track) * (1 - self.__thresh__)
            print(f"New ema is {new_ema}, while threshold is {threshold}")
      

        self.count += 1
        self.ema_track.append(new_ema)

        if self.count < self.__minimum_iters__:
            print("Still under minimum iterations threshold")
            return False
        
        if new_ema < threshold:
            return True
        else:
            return False

    def reset(self):
        self.auc_track = []
        self.ema_track = []
        self.count = 0


test = EarlyStopFlag(.05, .3, 10)
for i in range(0, 50):
    if test.update(np.random.randn()):
        break


torch.manual_seed(12345)
dataset = datasets.c1Data().shuffle()

train_dataset = dataset[:24]
test_dataset = dataset[24:]

# Batching
from torch_geometric.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

trainer = GCN_Train(33, 2, 16)
trainer.train(train_loader)

