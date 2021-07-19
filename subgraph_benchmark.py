from torch_geometric.data import dataset
import datasets
import torch
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from random import randint

# Build the GNN (Using the same architecture as from our first functional GNN)

# Training function
def train():
    model.train()

    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Test function
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim = 1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# Original c1 data
print("Dataset: C1 Full-graph")
dataset = datasets.c1Data()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
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

# Splitting dataset
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:24]
test_dataset = dataset[24:]

# Batching
train_loader = DataLoader(train_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Instantiate model
model = GCN(hidden_channels = 16).double()
print(model)

# Train the model
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

tracker1 = []

print(f"Learning rate = {lr}")
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker1.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


# Redo with a smaller learning rate.
model = GCN(hidden_channels = 16).double()
print(model)

# Train the model
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

tracker2 = []

print(f"Learning rate = {lr}")
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker2.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


# c1 data subgraphed by cell with highest CK expression and neighborhood of radius 5
print("Dataset: C1 CK Subgraphs")
dataset = datasets.c1_ck_subgraphs()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
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

# Construct the model
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker3 = []
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker3.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Redo but with a smaller learning rate
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker4 = []
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker4.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# c1 data subgraphed by area
print("Dataset: C1 Area Subgraphs")
dataset = datasets.c1_area_subgraphs()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
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

torch.manual_seed(12345)
test_index = []
for j in [j for j in [randint(0, len(dataset)/13) for i in range(0, 4)]]:
    test_index.extend([i for i in range(10 * j, 10 * j+10)])
train_index = [i for i in range(0, len(dataset)) if i not in test_index]

test_dataset = dataset[test_index]
train_dataset = dataset[train_index]

print(f"Number of test graphs: {len(test_dataset)}")
print(f"Number of training graphs: {len(train_dataset)}")

# Batching
train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 10, shuffle = False)

# Construct the model
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker5 = []
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker5.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Redo but with a smaller learning rate
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker6 = []
for epoch in range(1, 51):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker6.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

#### Visualization of results
import matplotlib.pyplot as plt

# Original c1, lr = 0.01
train_line, = plt.plot([x[0] for x in tracker1], [x[1] for x in tracker1], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker1], [x[2] for x in tracker1], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 Data, LR = 0.01')
plt.savefig('c1whole_lr01.png')
plt.clf()

# Original c1, lr = 0.005 
train_line, = plt.plot([x[0] for x in tracker2], [x[1] for x in tracker2], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker2], [x[2] for x in tracker2], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 Data, LR = 0.005')
plt.savefig('c1whole_lr005.png')
plt.clf()

# Subgraph by CK, lr = 0.01
train_line, = plt.plot([x[0] for x in tracker3], [x[1] for x in tracker3], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker3], [x[2] for x in tracker3], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.01')
plt.savefig('c1ck_lr01.png')
plt.clf()

# Subgraph by CK, lr = 0.05
train_line, = plt.plot([x[0] for x in tracker4], [x[1] for x in tracker4], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker4], [x[2] for x in tracker4], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.005')
plt.savefig('c1ck_lr005.png')
plt.clf()

# Subgraph by area, lr = 0.01
train_line, = plt.plot([x[0] for x in tracker5], [x[1] for x in tracker5], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker5], [x[2] for x in tracker5], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 area subgraphs, LR = 0.01')
plt.savefig('c1area_lr01.png')
plt.clf()

# Subgraph by area, lr = 0.05
train_line, = plt.plot([x[0] for x in tracker6], [x[1] for x in tracker6], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker6], [x[2] for x in tracker6], label = "Test Accuracy")
plt.axis([1, 50, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 area subgraphs, LR = 0.005')
plt.savefig('c1area_lr005.png')
plt.clf()

# Trying with significantly more epochs / smaller LR
dataset = datasets.c1_ck_subgraphs()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
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

# Construct the model
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker7 = []
for epoch in range(1, 1001):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker7.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

train_line, = plt.plot([x[0] for x in tracker7], [x[1] for x in tracker7], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker7], [x[2] for x in tracker7], label = "Test Accuracy")
plt.axis([1, 1000, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.001')
plt.savefig('c1ck_lr001.png')
plt.clf()
# So we're egregiously overfitting, but getting occasional peaks in the test accuracy. 
# I'm going to try using a much smaller learning rate and seeing how that affects it. 

# Small LR, big epoch
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

# Construct the model
model = GCN(hidden_channels=16).double()
print(model)

# Training the model
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = torch.nn.CrossEntropyLoss()

print(f"Learning rate = {lr}")
tracker8 = []
for epoch in range(1, 5001):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    tracker8.append([epoch, train_acc, test_acc])
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

train_line, = plt.plot([x[0] for x in tracker8], [x[1] for x in tracker8], label = "Training Accuracy")
test_line, = plt.plot([x[0] for x in tracker8], [x[2] for x in tracker8], label = "Test Accuracy")
plt.axis([1, 5000, 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training Accuracy", 'Test Accuracy'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.0001')
plt.savefig('c1ck_lr0001.png')
plt.clf()
# Smaller LR hits a much higher test accuracy that plateaus around 500 epochs. 
# would it be neneficial to continue tuning the LR? Or perhaps implementing a CV strategy here would be better. 
# Could also consider what happens if we don't restrict our datasplits along patient identity. 
