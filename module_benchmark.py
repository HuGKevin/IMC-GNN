import datasets as ds
import matplotlib.pyplot as plt
import models
import torch
from torch_geometric.data import DataLoader
from random import randint

output_folder = "GNN_Train"

torch.manual_seed(12345)
dataset = ds.c1Data().shuffle()

train_dataset = dataset[:24]
test_dataset = dataset[24:]

# Batching
train_loader = DataLoader(train_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

trainer = models.GCN_Train(33, 2, 16, es_min_iter=100)
while not trainer.flagraiser:
    trainer.train(train_loader, verbose = True)
    trainer.validate(test_loader, verbose = True)

# Running this on the original dataset, it maxes out performance on the training set way too fast, and the validation AUC plateaus at 0.5 because there no longer any loss. 
training_auc = trainer.train_acc
valid_auc = trainer.flag.auc_track

train_line, = plt.plot([i for i in range(1, len(training_auc) + 1)], training_auc, label = "Training AUC")
test_line, = plt.plot([i for i in range(1, len(valid_auc) + 1)], valid_auc, label = "Validation AUC")
plt.axis([1, len(training_auc), 0, 1])
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend([train_line, test_line], ["Training AUC", 'Test AUC'])
plt.title('GNN Performance on c1 graphs, LR = 0.001')
plt.savefig(f'{output_folder}/c1_graphs.png')




# c1 CK subgraphs
torch.manual_seed(12345)
dataset = ds.c1_ck_subgraphs()

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

c1_trainer = models.GCN_Train(33, 2, 16, es_min_iter=100)
while not c1_trainer.flagraiser:
    c1_trainer.train(train_loader, verbose = True)
    c1_trainer.validate(test_loader, verbose = True)

c1_training_auc = c1_trainer.train_acc
c1_valid_auc = c1_trainer.flag.auc_track

plt.clf()
train_line, = plt.plot([i for i in range(1, len(c1_training_auc) + 1)], c1_training_auc, label = "Training AUC")
test_line, = plt.plot([i for i in range(1, len(c1_valid_auc) + 1)], c1_valid_auc, label = "Validation AUC")
plt.axis([1, len(c1_training_auc), 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training AUC", 'Validation AUC'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.001')
plt.savefig(f'{output_folder}/c1_ck_subgraphs.png')

# Trying with a smaller minimum iterations
c1_trainer = models.GCN_Train(33, 2, 16, es_min_iter=10)
while not c1_trainer.flagraiser:
    c1_trainer.train(train_loader, verbose = True)
    c1_trainer.validate(test_loader, verbose = True)

c1_training_auc = c1_trainer.train_acc
c1_valid_auc = c1_trainer.flag.auc_track
plt.clf()
train_line, = plt.plot([i for i in range(1, len(c1_training_auc) + 1)], c1_training_auc, label = "Training AUC")
test_line, = plt.plot([i for i in range(1, len(c1_valid_auc) + 1)], c1_valid_auc, label = "Validation AUC")
plt.axis([1, len(c1_training_auc), 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training AUC", 'Validation AUC'])
plt.title('GNN Performance on c1 CK subgraphs, LR = 0.001')
plt.savefig(f'{output_folder}/c1_ck_subgraphs_less.png')



# With area subgraphs
area_dataset = ds.c1_area_subgraphs()

torch.manual_seed(12345)
test_index = []
for j in [j for j in [randint(0, len(area_dataset)/13) for i in range(0, 4)]]:
    test_index.extend([i for i in range(10 * j, 10 * j+10)])
train_index = [i for i in range(0, len(area_dataset)) if i not in test_index]

test_dataset = area_dataset[test_index]
train_dataset = area_dataset[train_index]

print(f"Number of test graphs: {len(test_dataset)}")
print(f"Number of training graphs: {len(train_dataset)}")

# Batching
train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 10, shuffle = False)

area_trainer = models.GCN_Train(33, 2, 16, es_min_iter=50)
while not area_trainer.flagraiser:
    area_trainer.train(train_loader, verbose = True)
    area_trainer.validate(test_loader, verbose = True)

area_training_auc = area_trainer.train_acc
area_valid_auc = area_trainer.flag.auc_track

# Stopped after 521 epochs because it looked like it was plateauing

# Plot training progress
plt.clf()
train_line, = plt.plot([i for i in range(1, len(area_training_auc) + 1)], area_training_auc, label = "Training AUC")
test_line, = plt.plot([i for i in range(1, len(area_valid_auc) + 1)], area_valid_auc, label = "Validation AUC")
plt.axis([1, len(area_training_auc), 0, 1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([train_line, test_line], ["Training AUC", 'Validation AUC'])
plt.title('GNN Performance on c1 area subgraphs, LR = 0.001')
plt.savefig(f'{output_folder}/c1_area_subgraphs.png')