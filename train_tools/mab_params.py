import torch
import torch.nn as nn
import random
import copy
import numpy as np

__all__ = ['init_mab_params', 'decay_mab_params', 'get_prior_mab_params', 'update_mab_params']


def init_mab_params(algo, clients):
    mab_params = {}
    if 'ucb' in algo:
        mab_params['reward_mean'] = dict(zip(clients, [np.random.binomial(1, 0.5) for _ in clients])) 
        mab_params['n'] = dict(zip(clients, [1 for _ in clients]))  
        mab_params['rounds'] = len(clients)
        
    elif 'thompson' in algo:
        mab_params['alpha'] = dict(zip(clients, [1 for _ in clients]))  
        mab_params['beta'] = dict(zip(clients, [1 for _ in clients]))  
        
    else:
        raise NotImplemented

    return mab_params


# only for thompson sampling algorithm
def decay_mab_params(mab_params, args):
    for c in mab_params['alpha'].keys():
        mab_params['alpha'][c] *= args.mab_decay
        mab_params['beta'][c] *= args.mab_decay
        
    return mab_params


def get_prior_mab_params(algo, mab_params, pretrained_validity):
    # How to use pretrained_validity to get mab prior?
    if 'ucb' in algo:
        pass
    elif 'thompson' in algo:
        pass
    else:
        raise NotImplemented


# deleted function
# def _singular_based_reward(model, server_weight, prev_server_weight, selected_clients, args, prev_info=None):
#     last_layer = list(server_weight.keys())[-1] if 'weight' in list(server_weight.keys())[-1] else list(server_weight.keys())[-2]
    
#     if prev_info is not None:
#         prev_sing = prev_info
#     else:
#         prev_server_fc = prev_server_weight[last_layer].data
#         _, prev_sing, _ = torch.svd(prev_server_fc)
    
#     updated_server_fc = server_weight[last_layer].data
#     _, updated_sing, _ = torch.svd(updated_server_fc)
    
#     # discrete reward : geometrical mean
#     reward = 1 if (prev_sing[args.num_classes-1] / prev_sing[0]) < (updated_sing[args.num_classes-1] / updated_sing[0]) else 0
    
#     return reward, updated_sing


def _get_statistics(model, valid_loader, args):
    prediction = lambda logits : torch.argmax(logits, 1)
    criterion = nn.CrossEntropyLoss()
    
    loss, acc = 0.0, 0.0
    
    model.eval()
    for i, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            loss += criterion(logits, labels).item()
            acc += (prediction(logits) == labels.data).sum().item()

    return loss, acc
    
    
def _valid_based_reward(model, valid_loader, server_weight, prev_server_weight, selected_clients, args, prev_info=None):
    
    # logit-ensemble (neurips 2020)
    if prev_info is not None:
        prev_acc, prev_loss = prev_info
    else:
        model.load_state_dict(prev_server_weight)
        prev_loss, prev_acc =  _get_statistics(model, valid_loader, args)
    model.load_state_dict(server_weight)
    updated_loss, updated_acc =  _get_statistics(model, valid_loader, args)
    
    print(prev_acc, updated_acc)
    if args.reward == 'val-loss':
        reward = 1 if prev_loss > updated_loss else 0
    else:
        reward = 1 if prev_acc < updated_acc else 0
    
    return reward, [updated_acc, updated_loss]

    
# deleted function
# def _bias_based_reward(model, server_weight, prev_server_weight, delta_dict, selected_clients, args, prev_info=None):
#     last_layer = list(server_weight.keys())[-1] if 'bias' in list(server_weight.keys())[-1] else list(server_weight.keys())[-2]
    
#     delta_fc = delta_dict[last_layer].data
#     bias_var = torch.var(delta_fc)
    
#     # for the first step
#     if prev_info is None:
#         reward = 1
#     else:
#         # discrete reward : geometrical mean
#         reward = 1 if bias_var < prev_info else 0
    
#     return reward, bias_var


def _get_arm_reward(model, client_loader, server_weight, prev_server_weight, delta_dict, selected_clients, args, prev_info=None):
    if args.reward == 'val-loss' or args.reward == 'val-acc':
        reward, prev_info = _valid_based_reward(model, client_loader['valid'], server_weight, prev_server_weight, selected_clients, args, prev_info=prev_info)
#     elif args.reward == 'singular':
#         reward, prev_info = _singular_based_reward(model, server_weight, prev_server_weight, selected_clients, args, prev_info=prev_info)
#     elif args.reward == 'bias':
#         reward, prev_info = _bias_based_reward(model, server_weight, prev_server_weight, delta_dict, selected_clients, args, prev_info=prev_info)
    else:
        raise NotImplemented
        
    return reward, prev_info
        

def update_mab_params(mab_params, model, client_loader, server_weight, prev_server_weight, delta_dict, selected_clients, args, prev_info=None):
    algo = args.algorithm
    
    # get reward for selected_clients
    reward, prev_info = _get_arm_reward(model, client_loader, server_weight, prev_server_weight, delta_dict, selected_clients, args, prev_info=prev_info)
    
    if not args.combination_bandit:
        for c in selected_clients:
            if 'ucb' in algo:
                mab_params['reward_mean'][c] *= mab_params['n'][c]
                mab_params['reward_mean'][c] += reward
                mab_params['n'][c] += 1
                mab_params['reward_mean'][c] /= mab_params['n'][c]

                mab_params['rounds'] += 1

            elif 'thompson' in algo:
                if reward == 1:
                    mab_params['alpha'][c] += 1
                else:
                    mab_params['beta'][c] += 1
    else:
        c = selected_clients
        if 'ucb' in algo:
            mab_params['reward_mean'][c] *= mab_params['n'][c]
            mab_params['reward_mean'][c] += reward
            mab_params['n'][c] += 1
            mab_params['reward_mean'][c] /= mab_params['n'][c]

            mab_params['rounds'] += 1

        elif 'thompson' in algo:
            if reward == 1:
                mab_params['alpha'][c] += 1
            else:
                mab_params['beta'][c] += 1
                
    # decay mab thompson parameters
    if 'thompson' in args.algorithm and args.mab_decay != 1.0:
        mab_params = decay_mab_params(mab_params, args)
        
    return mab_params, prev_info
