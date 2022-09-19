import os, sys
import copy
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import *
from data_utils import *
from models import *
from train_tools.criterion import *
from train_tools.mab_params import init_mab_params, get_prior_mab_params
from train_tools.client_opt import client_opt
from train_tools.server_opt import server_opt
from utils.device import *
from utils.util import *

MODEL = {'lenet': LeNet, 'lenetcontainer': LeNetContainer, 'vgg11': vgg11,
         'vgg11-bn': vgg11_bn, 'vgg13': vgg13, 'vgg13-bn': vgg13_bn,
         'vgg16': vgg16, 'vgg16-bn': vgg16_bn, 'vgg19': vgg19, 'vgg19-bn': vgg19_bn,
         'resnet8': resnet8, 'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
         'resnet20': resnet20, 'resnet32': resnet32, 'resnet44': resnet44, 'resnet56': resnet56, 'resnet110': resnet110, 'resnet1202': resnet1202,
         'simpleNet3': simpleNet3, 'simpleNet4': simpleNet4, 'simpleNet5': simpleNet5,
         'simpleNet6': simpleNet6, 'simpleNet7': simpleNet7, 'simpleNet4_bn': simpleNet4_bn, 'simpleNet4_sc':simpleNet4_sc}

DATASET = {'dirichlet-mnist': dirichlet_data_loader, 'dirichlet-cifar10': dirichlet_data_loader,
           'dirichlet-fmnist': dirichlet_data_loader, 'dirichlet-cifar100': dirichlet_data_loader,
           'emnist': load_partition_data_federated_emnist, 'fed-cifar100': load_partition_data_federated_cifar100,
           'synthetic': load_partition_data_federated_synthetic_1_1,'landmark-g23k': load_partition_data_landmarks_g23k,
           'landmark-g160k': load_partition_data_landmarks_g160k}

CRITERION = {'ce': cross_entropy, 'elr': elr_loss, 'elr+distil': distil_elr_loss}

if __name__ == '__main__':
    args = parse_args()
    args = align_args(args)
    
    # make experiment reproducible
    fix_seed(args.seed)
        
    # model log file
    log_filename, log_pd, check_filename, img_filename = make_log(args)
    
    # create dataloader
    if 'dirichlet' in args.dataset:
        client_loader, dataset_sizes, class_num, num_clients = DATASET[args.dataset](args, img_filename)
    else:
        client_loader, dataset_sizes, class_num, num_clients = DATASET[args.dataset](args.data_dir, args.batch_size)
    
    # data setting
    args.num_classes, args.num_clients = class_num, num_clients
    log_filename, check_filename, img_filename = modify_path(args)
    
    # create model for server and client
    model = MODEL[args.model](num_classes=args.num_classes, **args.model_kwargs) if args.model_kwargs else MODEL[args.model](num_classes=args.num_classes)
    
    # model to gpu
    model = model.to(args.device)
        
    # initialize server and client model weights
    server_weight = gpu_to_cpu(copy.deepcopy(model.state_dict()))

    server_momentum = {}
    client_weight = {}
    client_momentum = {}
    for client in range(args.num_clients):
        client_weight[client] = gpu_to_cpu(copy.deepcopy(model.state_dict()))
        client_momentum[client] = {}

    pretrained_validity = None
    prev_reward, prev_valid = None, None
    
    # individual client update
    if args.combination_bandit:
        client_sets = list(itertools.combinations(client_loader['train'].keys(),  int(float(eval(args.clients_per_round)))))
    else:
        client_sets = client_loader['train'].keys()
        
    if 'mab' in args.algorithm:
        mab_params = init_mab_params(args.algorithm, client_sets)
        if args.prior_params: mab_params = get_prior_mab_params(args.algorithm, mab_params, pretrained_validity)
    else:
        mab_params = None
    
    # how many time each client was selected during progress
    selected_clients_num = np.zeros(args.num_clients)
    
    if 'elr' in args.local_criterion:
        criterion = CRITERION[args.local_criterion](sum(dataset_sizes['train'].values()), class_num, args)
    else:
        criterion = CRITERION[args.local_criterion]

    best_acc = [-np.inf, 0, 0, 0, 0]
    test_acc = []
    best_fig = None
    best_weight = {'server': copy.deepcopy(server_weight), 'client': copy.deepcopy(client_weight)}

    # Federated Learning Pipeline
    for r in tqdm(range(args.num_rounds)):
        # client selection and updated the selected clients
        client_weight, selected_clients, client_momentum = client_opt(model, client_loader['train'], dataset_sizes['train'], server_weight, client_weight, client_momentum, criterion, args, rounds=r, mab_params=mab_params)
        
        # aggregate the updates and update the server
        model, server_weight, server_momentum, _, client_weight = server_opt(model, client_loader['train'], dataset_sizes['train'], server_weight, server_momentum, client_weight, selected_clients, args, iterRound=r, mab_params=mab_params, prev_reward=prev_reward, prev_valid=prev_valid)
        
        # Now, model has the round r server weights!
        
        # init mab params and client_momentum when decaying lr   
        if (r+1) in [int(epo) for epo in args.milestones.split(',')]: 
            if 'mab' in args.algorithm:
                mab_params = init_mab_params(args.algorithm, client_sets)
                if args.prior_params: mab_params = get_prior_mab_params(args.algorithm, mab_params, pretrained_validity)
        
        # update the history of selected_clients
        for oc in selected_clients: selected_clients_num[oc] += 1
        
        # evaluate the generalization of the server model
        acc, std, min, max, ece, fig = evaluate_accuracy(model, dataset_sizes['test'], client_loader['test'], args)
        
        print('Test Accuracy: %.2f' % acc)
        test_acc.append(acc)

        memchk_val = []
        if args.memorization_check:
            memchk_val = list(memorization_check(model, server_weight, dataset_sizes['check'], client_loader['check'], args))[1:]
        
        client_acc = []
        client_acc = ['-'] * args.client_accuracy * (1 + args.memorization_check) * args.num_clients
        
        log_pd.loc[r] = [acc, std, min, max] + ece + memchk_val + client_acc
        # log_pd.to_csv(log_filename)        
        
        if acc >= best_acc[0]:
            best_acc = [acc, std, min, max] + ece  + memchk_val
            best_fig = fig
            best_weight['server'] = copy.deepcopy(server_weight)
            best_weight['client'] = copy.deepcopy(client_weight)
        if mab_params is not None: mab_plotter(mab_params, r, img_filename, args)
        if r % 10 == 9:
            check_point_weight = {}
            check_point_weight['server'] = copy.deepcopy(server_weight)
            check_point_weight['client'] = copy.deepcopy(client_weight)
            # if not os.path.isdir(check_filename.replace('.t1', '') + '/'):
            #     os.makedirs(check_filename.replace('.t1', '') + '/')

            # torch.save(check_point_weight, check_filename.replace('.t1', '') + f'/epoch={r+1}.t1')
    #selected_clients_plotter(selected_clients_num, img_filename, args)
    #test_acc_plotter(test_acc, img_filename, args)
    
    # save file
    # state={}
    # state['final'] = server_weight
    # state['best'] = best_weight['server']
    # state['model'] = model
    # state['best_client'] = best_weight['client']
    # torch.save(state, check_filename)
    
    print('Successfully saved' + check_filename)
    print('Best Test Accuracy: %.2f' % best_acc[0])
    
    log_filename = log_filename.replace(".csv", "_best.csv")
    client_acc = personalized_accuracy(model, best_weight['server'], dataset_sizes, client_loader, class_num, args)
    
    log_pd.loc[args.num_rounds] = best_acc + client_acc
    log_pd.to_csv(log_filename)
    
    if best_fig: best_fig.savefig(img_filename, bbox_inches = 'tight')
