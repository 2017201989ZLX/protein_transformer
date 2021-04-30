from utils.utils_profiling import * # load before other local modules

import argparse
import os
import sys
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import dgl
import math
import numpy as np
import torch
import wandb
import json

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from protein_data import proteinDataset

from experiments.qm9.models import SE3Transformer

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()
    train_pair = json.load(open('experiments/qm9/only_train_pair.json','r'))
    num_iters = len(dataloader)
    for i, (g1,g2, a1,a2,y) in enumerate(dataloader):
        if g1 == None:
            continue
        g1 = g1.to(FLAGS.device)
        g2 = g2.to(FLAGS.device)
        y = y.to(FLAGS.device)
        

        optimizer.zero_grad()

        # run model forward and compute loss
        atom1 = []
        atom2 = []
        label = []
        atom_bias1 = 0
        atom_bias2 = 0
        cnt = 0
        for i in y:
            train_pair[i] = np.array(train_pair[i])
#             print(train_pair[i][:,0],atom_bias1)
            atom1.extend(train_pair[i][:,0]+atom_bias1)
            atom2.extend(train_pair[i][:,1]+atom_bias2)
            label.extend(train_pair[i][:,2])
            atom_bias1 += a1[cnt].numpy()
            atom_bias2 += a2[cnt].numpy()
            cnt += 1
        pred = model(g1,g2,atom1,atom2)
        l1_loss, __ = loss_fnc(pred, torch.tensor(label).float().to(FLAGS.device))

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} [units]")
#             print(';;;;')
        if i % FLAGS.log_interval == 0:
            wandb.log({"Train L1 loss": to_np(l1_loss)})

        if FLAGS.profile and i == 10:
            sys.exit()
    
        scheduler.step(epoch + i / num_iters)

def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()
    train_pair = json.load(open('experiments/qm9/only_test_pair.json','r'))
    rloss = 0
    auc_hit = []
    for i, (g1,g2, a1,a2,y) in enumerate(dataloader):
        if g1 == None:
            continue
        g1 = g1.to(FLAGS.device)
        g2 = g2.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        atom1 = []
        atom2 = []
        label = []
        pair_index = [0]
        atom_bias1 = 0
        atom_bias2 = 0
        cnt = 0
        for i in y:
            train_pair[i] = np.array(train_pair[i])
#             print(train_pair[i])
#             print(train_pair[i][:,0],atom_bias1)
            atom1.extend(train_pair[i][:,0]+atom_bias1)
            atom2.extend(train_pair[i][:,1]+atom_bias2)
            label.extend(train_pair[i][:,2])
            pair_index.append(len(train_pair[i]))
            atom_bias1 += a1[cnt].numpy()
            atom_bias2 += a2[cnt].numpy()
            cnt += 1

        # run model forward and compute loss
        pred = model(g1,g2,atom1,atom2).detach()
        for i in range(len(pair_index)-1):
            pred_po = pred[pair_index[i]:pair_index[i+1]].squeeze(1).cpu().detach().numpy()
            label_po = np.array(label[pair_index[i]:pair_index[i+1]])
            sorted_indices = np.argsort(-pred_po)
            hit_num = len(sorted_indices)
#             print(hit_num)
#             right_label = (label_po == 1)
            if np.sum(label_po) == 0:
                continue

            for i in range(len(sorted_indices)):
                if label_po[sorted_indices[i]] == 1:
                    hit_num = i
                    break
            auc_hit.append(hit_num/len(sorted_indices))
    print('test_result:',auc_hit,np.sum(auc_hit)/len(auc_hit))
#         __, __, rl = loss_fnc(pred, y, use_mean=False)
#         rloss += rl
#     rloss /= FLAGS.test_size

#     print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")
#     wandb.log({"Test L1 loss": to_np(rloss)})

def eval(epoch, model, loss_fnc, dataloader, FLAGS):
    model.load_state_dict(torch.load("models/E-d4-l4-16-0.pt"))
    model.to(FLAGS.device)
    print('load model already')
    test_pair = json.load(open('experiments/qm9/test_pairs.json','r'))
    train_pair = json.load(open('experiments/qm9/only_test_pair.json','r'))
    protein_logit = {}
    protein_label = {}
    auc_hit = []
    auc_pos = []
    auc_protein = []
    for i, (g1,g2, a1,a2,y) in enumerate(dataloader):
        if g1 == None:
            continue
        g1 = g1.to(FLAGS.device)
        g2 = g2.to(FLAGS.device)
        y = y.to(FLAGS.device)

        # run model forward and compute loss
        atom1 = []
        atom2 = []
        label = []
        pair_index = [0]
        atom_bias1 = 0
        atom_bias2 = 0
        pair_bias = 0
        cnt = 0
        for i in y:
            train_pair[i] = np.array(train_pair[i])
#             print(train_pair[i][:,0],atom_bias1)
            atom1.extend(train_pair[i][:,0]+atom_bias1)
            atom2.extend(train_pair[i][:,1]+atom_bias2)
            label.extend(train_pair[i][:,2])
            pair_index.append(len(train_pair[i])+pair_bias)
            atom_bias1 += a1[cnt].numpy()
            atom_bias2 += a2[cnt].numpy()
            pair_bias += len(train_pair[i])
            cnt += 1

        # run model forward and compute loss
        pred = model(g1,g2,atom1,atom2).detach()
        for i in range(len(pair_index)-1):
            pred_po = list(pred[pair_index[i]:pair_index[i+1]].squeeze(1).cpu().detach().numpy())
            label_po = label[pair_index[i]:pair_index[i+1]]
            if test_pair[y[i]]['potein'] not in protein_logit:
                protein_logit[test_pair[y[i]]['potein']] = []
            if test_pair[y[i]]['potein'] not in protein_label:
                protein_label[test_pair[y[i]]['potein']] = []
            protein_logit[test_pair[y[i]]['potein']].extend(pred_po)
            protein_label[test_pair[y[i]]['potein']].extend(label_po)
    
    for p in protein_logit:
        pred_po = np.array(protein_logit[p])
        label_po = np.array(protein_label[p])
        sorted_indices = np.argsort(-pred_po)
        hit_num = len(sorted_indices)
        for i in range(len(sorted_indices)):
            if label_po[sorted_indices[i]] == 1:
                hit_num = i
                break
#         print(len(sorted_indices),p)
        if hit_num == len(sorted_indices):
            print(p, len(label_po))
            continue
        auc_hit.append(hit_num/len(sorted_indices))
        auc_protein.append(p)
        auc_pos.append(hit_num)
    sorted_indices = np.argsort(auc_hit)
    for j in range(len(auc_hit)):
        i = sorted_indices[j]
        print(f"protein: {auc_protein[i]} score: {auc_hit[i]:.5f} hit_pos: {auc_pos[i]}")
    print('test_result:',auc_hit,np.sum(auc_hit)/len(auc_hit))

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def collate(samples):
    graphs1, graphs2, a1,a2,train_idx = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    return batched_graph1, batched_graph2,torch.tensor(a1),torch.tensor(a2),torch.tensor(train_idx)

    
def main(FLAGS, UNPARSED_ARGV):

    # Prepare data
    train_dataset = proteinDataset(mode='train', transform=RandomRotation())
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,shuffle=False, collate_fn=collate,num_workers=FLAGS.num_workers)


    test_dataset = proteinDataset( mode='test', transform=None) 
    test_loader = DataLoader(test_dataset, 
                             batch_size=FLAGS.batch_size, shuffle=False, 
                             collate_fn=collate, 
                             num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(train_dataset)
    FLAGS.test_size = len(test_dataset)

    # Choose model
    model = SE3Transformer(FLAGS.num_layers, 
                                             train_dataset.atom_feature_size, 
                                             FLAGS.num_channels,
                                             num_nlayers=FLAGS.num_nlayers,
                                             num_degrees=FLAGS.num_degrees,
                                             div=FLAGS.div,
                                             pooling=FLAGS.pooling,
                                             n_heads=FLAGS.head)
    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)
    #wandb.watch(model)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)

    # Loss function
    def task_loss(pred, target, use_mean=True):
        l1_loss = torch.sum(torch.abs(pred - target))
        l2_loss = torch.sum((pred - target)**2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]
        return l1_loss, l2_loss

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
#         test_epoch(epoch, model, task_loss, test_loader, FLAGS)
    eval(0, model, task_loss, test_loader, FLAGS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer', 
            help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
            help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
            help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16,
            help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0,
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=8, 
            help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=5000, 
            help="Number of epochs")

    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--log_interval', type=int, default=10,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=5,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='equivariant-attention', 
            help="wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(FLAGS.device)
    # Log all args to wandb
    if FLAGS.name:
        wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}')
    else:
        wandb.init(project=f'{FLAGS.wandb}')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    main(FLAGS, UNPARSED_ARGV)
