# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:03:19 2023

@author: luzy1
"""

#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_channel=1, out_channel=4):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1200, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(600, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True))
        
        self.fc4 = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True))
        
        
        self.fc5 = nn.Sequential(
            nn.Linear(500, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        
        self.fc6 = nn.Sequential(
            nn.Linear(128, 4),
            # nn.BatchNorm1d(4),
            # nn.ReLU(inplace=True)
            )
        



    def forward(self, x):


        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        
        return x
    
    
    
    
    def fea(self,x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x
        



