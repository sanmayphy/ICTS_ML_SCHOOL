import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

from typing import Optional, Tuple

from typing import Callable, Union
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch import Tensor

torch.set_default_dtype(torch.float64)

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
    PairTensor
)

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse, to_undirected
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, knn_graph
from torch_geometric.datasets import QM9
from torch_scatter import scatter
from torch_cluster import knn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import uproot
import vector
vector.register_awkward()
import awkward as ak

from IPython.display import HTML


class Jet_Dataset(data.Dataset):

    def __init__(self, dataset_path:str, tree_name:str = 'tree', k:int = 5) -> None:
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super(Jet_Dataset, self).__init__()
        
        
        self.dataset = uproot.open(dataset_path)
        self.tree = self.dataset[tree_name].arrays()
        
        self.num_entries = self.dataset[tree_name].num_entries
        
        self.part_feat = self.dataset[tree_name].keys(filter_name='part_*')
        self.jet_feat = self.dataset[tree_name].keys(filter_name='jet_*')
        self.labels = self.dataset[tree_name].keys(filter_name='labels_*')
        
        self.k = k
        
        
        #self.pc_dataset = [ self.transform_jet_to_point_cloud(idx) for idx in range(self.num_entries-1) ]
        

    def transform_jet_to_point_cloud(self, idx:int) -> Data :
    
        npart = self.tree['jet_nparticles'].to_numpy()[idx:idx+1]
        
        part_feat_list = [ak.flatten(self.tree[part_feat][idx:idx+1]).to_numpy() for part_feat in self.part_feat]
        
        jet_pt = self.tree['jet_pt'].to_numpy()[idx:idx+1]
        jet_eta = self.tree['jet_eta'].to_numpy()[idx:idx+1]
        jet_phi = self.tree['jet_phi'].to_numpy()[idx:idx+1]
        jet_energy = self.tree['jet_energy'].to_numpy()[idx:idx+1]
        jet_tau21 = self.tree['jet_tau2'].to_numpy()[idx:idx+1]/self.tree['jet_tau1'].to_numpy()[idx:idx+1]
        jet_tau32 = self.tree['jet_tau3'].to_numpy()[idx:idx+1]/self.tree['jet_tau2'].to_numpy()[idx:idx+1]
        jet_tau43 = self.tree['jet_tau4'].to_numpy()[idx:idx+1]/self.tree['jet_tau3'].to_numpy()[idx:idx+1]
        
        
        jet_sd_mass = self.tree['jet_sdmass'].to_numpy()[idx:idx+1]
        
        jet_feat = np.stack([jet_pt, jet_eta, jet_phi, jet_energy, jet_tau21, jet_tau32, jet_tau43]).T
              
        #jet_feat = np.repeat(jet_feat, int(npart), axis=0)
             
        part_feat = np.stack(part_feat_list).T
        
        total_jet_feat = part_feat #np.concatenate((part_feat, jet_feat), axis=-1)
        total_jet_feat[np.isnan(total_jet_feat)] = 0.
        
        #print(type(total_jet_feat), 'total_jet_feat shape : ', total_jet_feat.shape)
        
        jet_class = -1
        
        if(self.tree['label_QCD'].to_numpy()[idx:idx+1] == 1) : jet_class = 0
        
        if( (self.tree['label_Tbqq'].to_numpy()[idx:idx+1] == 1) or
            (self.tree['label_Tbl'].to_numpy()[idx:idx+1] == 1)) : jet_class = 2
        
        if( (self.tree['label_Zqq'].to_numpy()[idx:idx+1] == 1) or
            (self.tree['label_Wqq'].to_numpy()[idx:idx+1] == 1)) : jet_class = 0
        
        if( (self.tree['label_Hbb'].to_numpy()[idx:idx+1] == True) or
            (self.tree['label_Hcc'].to_numpy()[idx:idx+1] == True) or
            (self.tree['label_Hgg'].to_numpy()[idx:idx+1] == True) or
            (self.tree['label_H4q'].to_numpy()[idx:idx+1] == True) or
            (self.tree['label_Hqql'].to_numpy()[idx:idx+1] == True) ) : jet_class = 1
        
        part_eta = torch.tensor( ak.flatten(self.tree['part_deta'][idx:idx+1]).to_numpy() )
        part_phi = torch.tensor( ak.flatten(self.tree['part_dphi'][idx:idx+1]).to_numpy() )
        eta_phi_pos = torch.stack([part_eta, part_phi], dim=-1)
        
        edge_index = torch_geometric.nn.pool.knn_graph(x = eta_phi_pos, k = self.k)
        
        src, dst = edge_index
                
        part_del_eta = part_eta[dst] - part_eta[src]
        part_del_phi = part_phi[dst] - part_phi[src]
        
        part_del_R = torch.hypot(part_del_eta, part_del_phi).view(-1, 1) # -- why do we need this view function ? 
        
        data = Data(x=torch.tensor(total_jet_feat), edge_index=edge_index, edge_deltaR = part_del_R)
        data.label = torch.tensor([jet_class])
        data.sd_mass = torch.tensor(jet_sd_mass)
        data.global_data = torch.tensor(jet_feat)
        data.seq_length = torch.tensor(npart)
        
        return data    
        

    def __len__(self) -> int:
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.num_entries#len(self.pc_dataset)
    
    def __getitem__(self, idx:int) -> Data :
        # Return the idx-th data point of the dataset
    
        return self.transform_jet_to_point_cloud(idx)#self.pc_dataset[idx]#data_point, data_label

