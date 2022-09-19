import random
import numpy as np
from numpy.random import choice

__all__ = ['client_selection', 'combination_client_selection']

# How to address K-arm problem?
def client_selection(clients, num_clients, args, mab_params, client_datasize=None):
    if 'fedavg' in args.algorithm and 'dropout' not in args.algorithm:
        selected_clients = random.sample(clients, num_clients)
        
    elif args.algorithm == 'fedprox':
        weight_dist = [num / sum(client_datasize) for num in client_datasize]
        selected_clients = choice(range(len(clients)), num_clients, p=weight_dist)
        
    elif 'dropout' in args.algorithm:
        num_dropout_clients = np.random.binomial(len(clients),num_clients / len(clients))
        print(num_dropout_clients)
        selected_clients = random.sample(clients, num_dropout_clients)
        
    elif 'ucb' in args.algorithm:        
        sampled_reward = []
        for c in clients:
            sampled_reward.append(mab_params['reward_mean'][c] + np.sqrt(3 * np.log(mab_params['rounds']) / (2 * mab_params['n'][c])))
            
        selected_clients = list(np.argsort(sampled_reward)[::-1][:num_clients])       
        
    elif 'thompson' in args.algorithm:
        sampled_reward = []
        for c in clients:
            sampled_reward.append(np.random.beta(mab_params['alpha'][c], mab_params['beta'][c]))
        selected_clients = list(np.argsort(sampled_reward)[::-1][:num_clients])
        
    else:
        raise NotImplemented
        
    return selected_clients


def combination_client_selection(clients, args, mab_params, client_datasize=None):
    if 'ucb' in args.algorithm:       
        best_sample = -np.inf
        for c in clients:
            sample = mab_params['reward_mean'][c] + np.sqrt(3 * np.log(mab_params['rounds']) / (2 * mab_params['n'][c]))
            
            if sample > best_sample:
                selected_clients = c
                best_sample = sample
            elif sample == best_sample:
                i = random.choice([0, 1])
                selected_clients = [selected_clients, c][i]
                best_sample = [best_sample, sample][i]
            else:
                continue
        
    elif 'thompson' in args.algorithm:
        best_sample = -np.inf
        for c in clients:
            sample = np.random.beta(mab_params['alpha'][c], mab_params['beta'][c])
            
            if sample > best_sample:
                selected_clients = c
                best_sample = sample
            elif sample == best_sample:
                i = random.choice([0, 1])
                selected_clients = [selected_clients, c][i]
                best_sample = [best_sample, sample][i]
            else:
                continue
    else:
        raise NotImplemented
        
    return selected_clients
