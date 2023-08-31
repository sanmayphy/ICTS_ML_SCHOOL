import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torch.nn as nn



def build_mlp(inputsize,outputsize,features,add_batch_norm=False,add_activation=None):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    layers.append(nn.ReLU())
    for hidden_i in range(1,len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i-1]))
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))
    if add_activation!=None:
        layers.append(add_activation)
    return nn.Sequential(*layers)
