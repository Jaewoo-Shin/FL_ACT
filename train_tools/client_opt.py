import copy
import torch
import itertools
from utils.device import *
from .criterion import *
from .criterion import elr_loss
from .optimizer import *
from .regularizer import *
from .client_selection import client_selection, combination_client_selection
from .client_scheduler import *

__all__ = ['client_opt']


OPTIMIZER = {'sgd': sgd, 'adam': adam, 'adagrad': adagrad}
CLSCHEDULER = {'multistep': mlst, 'exponential': exp, 'cosine': cos, 'uniform': uniform, 'linear': linear}


# train local clients
def client_opt(model, client_loader, client_datasize, server_weight, client_weight, client_momentum, criterion, args, rounds, mab_params=None, validation=False):
    # argument for training clients
    num_epochs = args.num_epochs
    
    optimizer = OPTIMIZER[args.local_optimizer]
    lr = scheduler(args.local_lr, rounds, args.lr_decay, [int(epo) for epo in args.milestones.split(',')], args.sch_type)
    
    # hyperparameters
    momentum = args.local_momentum
    nesterov = args.nesterov
    wd = args.wd
    mu = args.mu
    
    if validation:
        selected_clients = client_loader.keys()
    else:
        num_clients = CLSCHEDULER[args.cl_scheduler](round(float(eval(args.clients_per_round))), rounds, args)
        if args.combination_bandit:
            clients = list(itertools.combinations(client_loader.keys(),  int(float(eval(args.clients_per_round)))))
            selected_clients = combination_client_selection(clients, args, mab_params, client_datasize)
            print('[%s algorithm] %s clients are selected' % (args.algorithm, selected_clients))
        else:
            clients = client_loader.keys()
            selected_clients = client_selection(clients, num_clients, args, mab_params, client_datasize)
            print('[%s algorithm] %s clients are selected' % (args.algorithm, selected_clients))
        
    if args.teacher:
        teacher = copy.deepcopy(model)
        filename = './teacher_checkpoints/' + args.teacher
        teacher.cpu()
        teacher.load_state_dict(torch.load(filename,map_location='cpu')['final'])
        teacher = teacher.to(args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    if args.local_criterion == 'elr+distil':
        server_model = copy.deepcopy(model)
        server_weight = cpu_to_gpu(server_weight, args.device)
        server_model.load_state_dict(server_weight)
        server_weight = gpu_to_cpu(server_weight)
        server_model.eval()
        for param in server_model.parameters():
            param.requires_grad = False
    
    for client in set(selected_clients):
        # load model weights
        client_weight[client] = cpu_to_gpu(client_weight[client], args.device)
        model.load_state_dict(client_weight[client])
        if args.logit_distillation:
            tmp_model = copy.deepcopy(model)
            tmp_model.eval()
            for param in tmp_model.parameters():
                param.requires_grad = False
        
        # local training
        model.train()
        for epoch in range(num_epochs):
            for i, (indices, inputs, labels, gt_labels) in enumerate(client_loader[client]):
                # more develpment required
                if args.input_reg == None:
                    inputs, labels = input_reg(inputs, labels, args.device)
                
                elif args.input_reg == 'mixup':
                    lam, inputs, labels1, labels2 = input_reg(inputs, labels, args.device, mode=args.input_reg)
                
                if args.logit_distillation in ['mse', 'kl']:
                    tmp_pred = tmp_model(inputs)
                if args.teacher:
                    teacher_pred = teacher(inputs)

                # optimizer.zero_grad()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                
                with torch.set_grad_enabled(True):
                    # Forward pass
                    pred = model(inputs)
                    if args.teacher:
                        loss = kd_loss(pred,teacher_pred)
                    else:
                        if args.local_criterion == 'elr':
                            loss = criterion(pred, labels, indices)
                        elif args.local_criterion == 'elr+distil':
                            pred_s = server_model(inputs)
                            loss = criterion(pred, pred_s, labels)
                        else: #cross entropy
                            loss = criterion(pred, labels, args.num_classes, args.smoothing, args.temperature)

                    if args.logit_distillation=='mse' and rounds > 29:
                        loss = 0.5 * loss
                        loss += 0.5 * mse_logit(pred, tmp_pred)
                    elif args.logit_distillation == 'kl' and rounds > 29:
                        loss = 0.5 * loss
                        loss += 0.5 * kd_loss(pred, tmp_pred)
                        
                    # Backward pass (compute the gradient graph)
                    loss.backward()
                    
                    # regularization term
                    if wd:
                        model = weight_decay(model, wd)
                        
                    if mu:
                        server_weight = cpu_to_gpu(server_weight, args.device)
                        model = fedprox(model, mu, server_weight)
                        server_weight = gpu_to_cpu(server_weight)
                    
                    if momentum:
                        model, client_momentum = apply_local_momentum(model, momentum, nesterov, client_momentum, client, args.device)
                        
                    model = optimizer(model, lr)
                    
        # after local training
        client_weight[client] = gpu_to_cpu(copy.deepcopy(model.state_dict()))

    return client_weight, selected_clients, client_momentum
