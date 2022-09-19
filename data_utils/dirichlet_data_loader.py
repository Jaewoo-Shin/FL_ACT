import os
import copy
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import Dataset, DataLoader, sampler, Subset
from utils.util import fix_seed
from data_utils.BaseDataset import *

__all__ = ['dirichlet_data_loader']

def set_clients(_trainset, args, num_classes=10, data_filename=''):
    '''
    _trainset: whole train dataset
    num_clients: the number of clients. Should be integer type.
    replacement: boolean value whether sampling with replacement or not.
    dist_mode: 'client'-diri distribution for the number of clients. 'class'-diri distribution for the number of classes.
    
    Here, we handle the non-iidness via the Dirichlet distribution.
    non_iid: the concentration parameter alpha in the Dirichlet distribution. Should be float type.
    We refer to the paper 'https://arxiv.org/pdf/1909.06335.pdf'
    '''
    num_clients = args.num_clients
    non_iid = args.non_iid
    assert type(num_clients) is int, 'num_clients should be the type of integer.'
    assert type(non_iid) is float and non_iid > 0, 'iid should be the type of float.'
    
    # Generate the set of clients
    clients_dict = {}
    for i in range(num_clients):
        clients_dict[i] = []

    # Divide the dataset into each class of dataset.
    data_dict = {}
    for i in range(num_classes):
        data_dict[i] = []

    for idx, (_, _, label, gt_label) in enumerate(_trainset):
        data_dict[label].append(idx) #label or gt_label?
    
    data_dist = {}
    for client in range(num_clients):
        data_dist[client] = []
    
    # Distribute the data with the Dirichilet distribution.
    if args.dist_mode == 'class':
        fix_seed(args.seed) # to get the same class distribution for train and valid set
        diri_dis = torch.distributions.dirichlet.Dirichlet(non_iid * torch.ones(num_classes))
        remain = np.inf
        nums = len(_trainset) // num_clients
        while remain != 0: ######################################### too vulnerable
            for client_key in clients_dict.keys():
                if len(clients_dict[client_key]) >= nums:
                    continue
                
                tmp = diri_dis.sample()
                for cls in data_dict.keys():
                    tmp_set = random.sample(data_dict[cls], min(len(data_dict[cls]), int(nums * tmp[int(cls)])))
                    
                    if len(clients_dict[client_key]) + len(tmp_set) > nums:
                        tmp_set = tmp_set[:nums-len(clients_dict[client_key])]
                    clients_dict[client_key] += tmp_set
                    
                    if len(data_dist[client_key]) < len(data_dict):
                        data_dist[client_key].append(len(tmp_set))  
                    else:
                        data_dist[client_key][int(cls)] += len(tmp_set)
                    data_dict[cls] = list(set(data_dict[cls])-set(tmp_set))
                        
            remain = sum([len(d) for _, d in data_dict.items()])
            
        print('data_dist', [sum(data_dist[k]) for k in data_dist.keys()])
        #if num_classes == 10:
        #    data_plotter(data_dist, data_filename)

    return clients_dict

def dirichlet_data_loader(args, data_filename):
    _trainset = BaseDataset(args, train=True, noisy=True)
    _testset = BaseDataset(args, train=False, noisy=False)
    class_num = _trainset.classes

    clients_dict = {'train': {}, 'valid': {}}
    clients_dict['train'] = set_clients(_trainset, args, num_classes=class_num, data_filename=data_filename)
    clients_dict['valid'] = set_clients(_testset, args, num_classes=class_num, data_filename=data_filename)
    
    # client_loader: data loader of each client. type is dictionary
    client_loader = {'train': {}, 'valid': {}, 'test': {}, 'check': {}}
    dataset_sizes = {'train': {}, 'valid': {}, 'test': 0, 'check': 0}
    
    for client in clients_dict['train'].keys():
        client_dataset = Subset(_trainset, clients_dict['train'][client])
        client_loader['train'][client] = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
        dataset_sizes['train'][client] = len(client_dataset)
        
    for client in clients_dict['valid'].keys():
        client_dataset = Subset(_testset, clients_dict['valid'][client])
        client_loader['valid'][client] = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
        dataset_sizes['valid'][client] = len(client_dataset)
    
    client_loader['test'] = DataLoader(_testset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataset_sizes['test'] = len(_testset)

    client_loader['check'] = DataLoader(_trainset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataset_sizes['check'] = len(_trainset)
    
    return client_loader, dataset_sizes, class_num, args.num_clients