import torch.nn as nn
import math
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, nin, nhid):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(nin, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, 2)
    def forward(self, x, feature=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feature_map = F.relu(self.fc3(x))
        out = self.fc4(feature_map)

        if feature:
            return out, feature_map
        return out

def fnn(nin=2, nhid=500):
    return FNN(nin, nhid)