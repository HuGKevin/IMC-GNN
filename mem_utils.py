import torch
import matplotlib.pyplot as plt


def print_malloc_size(gpu):
    '''
    Simple print
    '''
    print(torch.cuda.memory_allocated(device)/1.e9)
    
def select_freest_device(device_pool):
    '''
    Selects first device in list device_pool that has enough memory (int mem_req)
    '''
    curr=0
    curr_dev = None
    for d in device_pool:
        dev_free = torch.cuda.memory_free(d)
        if curr<dev_free:
            curr_dev = d
            curr = int(dev_free)

    return curr_dev
    

class CUDAMemoryTracker():
    '''
    Class for memory debugging on python side, allows for use as callback
    '''
    
    def __init__(self, device):
        self.device = device
        self.mem_use = []
        self.labels = []
        self.curr_use = -1
        
    def reset(self):
        '''
        Reset this memory tracker
        '''
        self.mem_use = []
        self.labels = []
        self.curr_use = -1
        
    def add_mem_pt(self, label, verbose=False):
        '''
        Add a point for tracking memory, with label (may be used by callback, to track memory in 'hard to reach' locations)
        '''
        mem = torch.cuda.memory_allocated(self.device)/1.e9
        self.mem_use.append(mem)
        self.labels.append(label)
        self.curr_use = mem
        if verbose:
            print(label, ":", self.curr_use, " GB")
    
    def plot_mem_use(self, rang = None):
        '''
        Plot memory usage across all defined points. 'rang': tuple(int) defines x-range of plot
        '''
        if rang is None:
            rang = (0,len(self.mem_use))
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(self.mem_use[rang[0]:rang[1]])
        ax.set_xticks(np.arange(rang[1]-rang[0]))
        ax.set_xticklabels(self.labels[rang[0]:rang[1]], rotation=90)
        