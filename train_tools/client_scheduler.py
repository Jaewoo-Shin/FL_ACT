# client scheduling

# multistep
# exponential
# cosine
# warm-up

import numpy as np
import copy

__all__ = ['uniform', 'mlst', 'exp', 'cos', 'linear']

def mlst(init_clients, rounds, args):
    '''
    multistep scheduler for client selection.
    args.cl_decay, args.cl_milestones
    '''
    cl_milestones = [int(epo) for epo in args.cl_milestones.split(',')]

    num=0
    for i in range(len(cl_milestones)):
        if (rounds+1) > cl_milestones[i]:
            if i == len(cl_milestones)-1:
                num = i+1
                return round(init_clients * (args.cl_decay ** num))
            else:
                num = i + 1
        else:
            print(round(init_clients * (args.cl_decay ** num)))
            return round(init_clients * (args.cl_decay ** num))
        
def linear(init_clients, rounds, args):
    '''
    linear scheduler for client selection.
    args.cl_decay, args.cl_decay_per_round
    '''
    
    return init_clients - args.cl_decay * ((rounds+1) // args.cl_decay_per_round)
    

def exp(init_clients, rounds, args):
    '''
    multistep scheduler for client selection.
    args.cl_decay, args.cl_decay_per_round
    '''
    
    return int(init_clients * (args.cl_decay ** ((rounds+1) // args.cl_decay_per_round)))

def cos(init_clients, rounds, args):
    '''
    multistep scheduler for client selection.
    arg.cl_period, args.cl_par_min
    init_clients are considered as the maximum number.
    '''
    
    
    return round(args.cl_par_min + (init_clients - args.cl_par_min) / 2 * (1+np.cos(rounds / args.cl_period * np.pi)))

def uniform(init_clients, rounds, args):
    '''
    uniform scheduler for client selection.
    '''
    
    return init_clients