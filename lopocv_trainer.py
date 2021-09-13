import numpy as np
import torch
from models import GCN, GCN_Train
import copy

from torch_geometric.data import Data, Dataset, DataLoader, batch, Subset

class LOPOCV():
    
    def __init__(self, model, dataset, devices, batch_size):
        
        self.model_trainers = dict()
        self.train_loaders = dict()
        self.test_loaders = dict()
        self.device_pool = devices
        self.original_model = model
        
        for patient,datum in enumerate(dataset):
            #Get test set
            test_set = datum
            
            #Get training indices for current lopo model
            train_ind = list(range(len(dataset)))
            train_ind = train_ind[:patient] + train_ind[patient+1:]
            
            train_set = torch.utils.data.Subset(dataset, train_ind)
            self.train_loaders[patient] = DataLoader(train_set, batch_size=batch_size, shuffle = True)
            self.test_loaders[patient] = DataLoader(test_set, batch_size=batch_size, shuffle = False)
            
                        
            #create model
            self.model_trainers[patient] = copy.deepcopy(self.model)
            
    
    def load_model_into_pos(self, file, patient):
        '''
        Loads model from file, (usually will be directory/lopo$(patient).mdl)
        '''
        ###TODO MODEL LOAD FUNC
        self.model_trainers[patient] = self.model_trainers[patient].load(file)
        
    def load_all_models_from_dir(self, direc):
        '''
        Load all models from directory, files in format lopo$(patient).mdl
        '''
        for patient in self.model_trainers:
            self.load_model_into_pos(direc + 'lopo'+str(patient)+'.mdl', patient)
    
    def train(self, save_dir = None, verbose=False):
        '''
        Train all models on appropriate data from train loaders
        '''
        if verbose:
            print("Training all models in LOPO regime...")
            if save_dir is not None:
                print("-saving all models to", save_dir, "with naming conv: lopo$(patient).mdl")
            
        for patient in self.train_loaders:
            if verbose:
                print("Leaving patient", patient, "out")
            
            if verbose:
                print("---Selecting device and transferring")
                
            #Pick GPU with largest amount of memory free
            selected_device = select_freest_device(self.device_pool)
            if verbose:
                print("---Transferred to device", selected_device)
            
            
            ###TODO: edit device transfer
            self.model_trainers[patient].model.to(selected_device)
            
            if verbose:
                print("---Starting Train")
                print()
                print()
            
            self.model_trainers[patient].train(self.train_loaders[patient], verbose = verbose)
            
            if verbose:
                print("---Removing model from device")
            
            
            ###TODO: edit device transfer
            self.model_trainers[patient].model.to('cpu')
            
            if verbose and save_dir is not None:
                print("---Saving model as lopo", patient, ".mdl")

            ###TODO: need model save code
            self.model_trainers[patient].save(save_dir + 'lopo' + patient + '.mdl')

    def validate(self, device=None, verbose=False, aggregate_func=None):
        '''
        Validate all models on their respective leave one out patients, and aggregate if needed
        '''
        scores = dict()
        for patient in self.test_loaders:
            if verbose:
                print("Leaving patient", patient, "out")
            
            #validate model
            scores[patient] = self.model_trainers[patient].validate(self.test_loaders[patient], verbose = verbose)
            
        #Aggregate (or don't) the scores and return
        if aggregate_func is None:
            return scores
        else:
            if verbose:
                print("Aggregating scores")
            return aggregate_func(scores)
    
    def predict(self, x, edge_index, device=None, verbose=False, aggregate_func=None):
        '''
        Predict using all models on their respective leave one out patients, and aggregate if needed
        '''
        preds = dict()
        for patient in self.test_loaders:
            if verbose:
                print("Leaving patient", patient, "out")
            #predict with model
            preds[patient] = self.model_trainers[patient].predict(x, edge_index, verbose = verbose)
            
        #Aggregate (or don't) the predictions and return
        if aggregate_func is None:
            return preds
        else:
            if verbose:
                print("Aggregating predictions")
            return aggregate_func(preds)