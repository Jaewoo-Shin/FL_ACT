import torch
import copy
from utils.device import *
from .optimizer import *
from .mab_params import update_mab_params
from .criterion import gradientNoiseImage

__all__ = ['server_opt']


def server_opt(model, client_loader, client_datasize, server_weight, server_momentum, client_weight, selected_clients, args, iterRound, mab_params=None, prev_reward=None, prev_valid=None):
    # argument for select clients
    momentum = args.global_momentum
    # only use nesterov True update
    nesterov = args.nesterov
    
    # before aggregation, store the last round weight for reward
    prev_server_weight = copy.deepcopy(server_weight)
    
    # initialize the trained results
    delta_weight = {}
    for client in client_loader:
        delta_weight[client] = {}
    
    # get delta_weight of each client
    for client in client_loader:
        if client in selected_clients:
            for name, p in client_weight[client].items():
                delta_weight[client][name] = (server_weight[name].data - p.data).cpu()
        else:
            for name in client_weight[client].keys():
                delta_weight[client][name] = torch.zeros_like(server_weight[name]).cpu()
                
    # get aggregate weights
    if args.full_part:
        total_datasize = sum(client_datasize.values())
        aggregate_weights = {}
        for client in client_loader:
            aggregate_weights[client] = client_datasize[client] / total_datasize
    
    else:
        if args.algorithm == 'fedprox':
            aggregate_weights = {}
            for client in selected_clients:
                if aggregate_weights.get(client):
                    aggregate_weights[client] += 1 / len(selected_clients)
                else:
                    aggregate_weights[client] = 1 / len(selected_clients)

        else:
            # fedavg_conv, mab algorithm
            total_datasize = 0
            for client in selected_clients:
                total_datasize += client_datasize[client]

            aggregate_weights = {}
            for client in selected_clients:
                aggregate_weights[client] = client_datasize[client] / total_datasize    
    
    # get delta weights
    delta_dict = {}
    for client, agg in aggregate_weights.items():
        for name, delta in delta_weight[client].items():
            if delta_dict.get(name) is None:
                delta_dict[name] = (delta.data * agg).to(args.device)
            else:
                delta_dict[name] += (delta.data * agg).to(args.device)
                
    # server momentum
    if momentum:
        delta_dict, server_momentum = apply_global_momentum(delta_dict, server_momentum, momentum, nesterov, args.device)
    
    # update server weights
    server_weight = cpu_to_gpu(server_weight, args.device)
    
    # for gradient-based noise image
    model.load_state_dict(server_weight)

    model = pseudo_sgd(model, delta_dict)
    
    # update server weight and clients weight
    server_weight = gpu_to_cpu(copy.deepcopy(model.state_dict()))
    for client in client_weight.keys():
        client_weight[client] = copy.deepcopy(server_weight)

    return model, server_weight, server_momentum, delta_dict, client_weight
