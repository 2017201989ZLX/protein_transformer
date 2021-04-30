import os
import sys

import dgl
import numpy as np
import torch
import json

from torch.utils.data import Dataset, DataLoader

from scipy.constants import physical_constants

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
DTYPE = np.float32
DTYPE_INT = np.int32


class proteinDataset(Dataset):

    def __init__(self, mode='train',transform=None): 
        self.train_graph = json.load(open('experiments/qm9/%s_pairs.json'%(mode),'r'))
        self.protein_feature = json.load(open('experiments/qm9/atom_feature_coordinate.json','r'))
        self.len = len(self.train_graph)
        self.atom_feature_size = 9
        self.transform = True
        
    def RandomRotation(self,x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)),data] = 1
        return one_hot
    
    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms ,coordinate):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        
        adjacency = {}
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j and self.euler_distance(coordinate[i],coordinate[j]) < 8:
                    adjacency[(i, j)] = 1

        # Convert to numpy arrays
        
        src = []
        dst = []
        w = []
        for edge, weight in adjacency.items():
            src.append(edge[0])
            dst.append(edge[1])
            w.append(weight)

        return np.array(src), np.array(dst), np.array(w)

    def euler_distance(self,a,b):
        return np.linalg.norm(a - b)

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to QM9 representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        x = self.unit_conversion[self.task] * x
        return x

    def get_graph(self,x,feature):
        

        num_atoms = len(x)

        
        # Create nodes
        src, dst, w = self.connect_fully(num_atoms,x)
        # Create graph
        G = dgl.DGLGraph()
        G.add_nodes(num_atoms)
        G.add_edge(src, dst)
        
        if self.transform:
            x = self.RandomRotation(x).astype(DTYPE)
        # Add node features to graph
        
        G.ndata['x'] = torch.tensor(x.astype(DTYPE)) #[num_atoms,3]
        G.ndata['f'] = torch.tensor(feature[:, :, None].astype(DTYPE))
        
        G.edata['d'] = torch.tensor(x[dst].astype(DTYPE) - x[src].astype(DTYPE)) #[num_atoms,3]
#         G.edata['w'] = torch.tensor(w.astype(DTYPE)) #[num_atoms,4]

        return G,num_atoms

    def __getitem__(self, idx):
        # Load node features
        protein = self.train_graph[idx]['potein']
        chainA, chainB = self.train_graph[idx]['chains'].split('_')
        g1_features = np.array(list(self.protein_feature[protein][chainA].values()))
        g2_features = np.array(list(self.protein_feature[protein][chainB].values()))
        G1,num_atoms1 = self.get_graph(g1_features[:,9:],g1_features[:,:9])
        G2,num_atoms2 = self.get_graph(g2_features[:,9:],g2_features[:,:9])

        return G1, G2,num_atoms1,num_atoms2, idx

def get_train_pair(mode='train'):
    train_graph = json.load(open('experiments/qm9/%s_pairs.json'%(mode),'r'))
    protein_feature = json.load(open('experiments/qm9/atom_feature_coordinate.json','r'))
    train_pairs = []
    print(len(train_graph))
    for idx in range(len(train_graph)):
        train_pairs.append(train_graph[idx]['train_pair'])
#         zeros = []
#         ones = []
#         new_pairs = []
#         for pairs in train_graph[idx]['train_pair']:
#             if pairs[2] == 0:
#                 zeros.append(pairs)
#             elif pairs[2] == 1:
#                 ones.append(pairs)
#         rate = len(ones)/(len(ones)+len(zeros))
#         if (len(ones)+len(zeros))
            
    print(len(train_pairs))
    json.dump(train_pairs, open('experiments/qm9/only_%s_pair.json'%(mode),'w'))


if __name__ == "__main__":
    get_train_pair('train')
#     def collate(samples):
#         graphs1, graphs2, a1,a2,train_idx = map(list, zip(*samples))
#         batched_graph1 = dgl.batch(graphs1)
#         batched_graph2 = dgl.batch(graphs2)
#         return batched_graph1, batched_graph2,torch.tensor(a1),torch.tensor(a2),torch.tensor(train_idx)
#     dataset = proteinDataset(mode='train')
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
#     for i, (g1,g2, a1,a2,y) in enumerate(dataloader):
#         print(i)