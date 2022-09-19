import torch
import copy
from utils.device import *

__all__ = ['sgd', 'adam', 'adagrad', 'pseudo_sgd', 'apply_local_momentum', 'apply_global_momentum', 'scheduler']


def pseudo_sgd(model, delta_dict, lr=1.0):
    for name, p in model.state_dict().items():
        if 'num_batches_tracked' in name:
            p.data.sub_(delta_dict[name].data.long())
        else:
            p.data.sub_((delta_dict[name].data).mul(lr))
            
    return model
    
    
def sgd(model, lr):
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.data.sub_(p.grad, alpha=lr)

    return model


def adam(model, lr):
    raise NotImplemented

    
def adagrad(model, lr):
    raise NotImplemented

    
def agnostic_fl():
    # projected stochastic gradients
    # return model, aggregate_weights
    raise NotImplemented


def apply_local_momentum(model, momentum, nesterov, client_momentum, client, device):
    if len(client_momentum[client]) == 0:
        for name, p in model.named_parameters():
            if p.grad is not None:
                buf = client_momentum[client][name] = torch.clone(p.grad).detach()
                if nesterov:
                    p.grad.add_(buf, alpha = momentum)
                else:
                    p.grad = buf
        client_momentum[client] = gpu_to_cpu(client_momentum[client])
    # in the course of training
    else:
        client_momentum[client] = cpu_to_gpu(client_momentum[client], device)
        for name, p in model.named_parameters():
            if p.grad is not None:
                buf = client_momentum[client][name]
                buf.mul_(momentum).add_(p.grad, alpha=1.0)
                if nesterov:
                    p.grad.add_(buf, alpha = momentum)
                else:
                    p.grad = buf
        client_momentum[client] = gpu_to_cpu(client_momentum[client])     
    return model, client_momentum


def apply_global_momentum(delta_dict, server_momentum, momentum, nesterov, device):
    for name, p in delta_dict.items():
        if name not in server_momentum:
            buf = server_momentum[name] = torch.clone(p.data).detach()
            if nesterov:
                p.data.add_(buf, alpha = momentum)
            else:
                p.data = buf
            server_momentum = gpu_to_cpu(server_momentum)
        else:
            server_momentum = cpu_to_gpu(server_momentum, device)
            buf = server_momentum[name]
            buf.mul_(momentum).add_(p.data, alpha=1.0)
            if nesterov:
                p.data.add_(buf, alpha = momentum)
            else:
                p.data = buf
            server_momentum = gpu_to_cpu(server_momentum)
            
    return delta_dict, server_momentum

    
def scheduler(lr, rounds, decay = 0.1, milestones=[50, 75], sch_type='uniform'):
    
    if sch_type == 'uniform':
        num=0

        for i in range(len(milestones)):
            if (rounds+1) > milestones[i]:
                num = i+1

        return lr * (decay ** num)
    elif sch_type == 'exp':
        return lr * (decay ** (rounds+1))
    

