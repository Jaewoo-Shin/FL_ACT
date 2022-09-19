import copy
import torch
import numpy as np
import random
from torch.nn import functional as F
from train_tools.client_opt import client_opt
from train_tools.criterion import *
from utils.device import *
from utils.calibration_check import *

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def prediction(logits):
    return torch.argmax(logits, 1)

def memorization_check(model, model_weight, dataset_size, client_loader, args):
    num_cl = 0 # number of clean labels
    num_nl = 0 # number of noisy labels
    cl_c = cl_w = 0 # clean labels - correct/wrong
    nl_c = nl_m = nl_w = 0 # noisy labels - correct/memorized/wrong
    model.load_state_dict(model_weight)
    model.eval()
    for i, (indices, inputs, labels, gt_labels) in enumerate(client_loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        gt_labels = gt_labels.to(args.device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            pred = prediction(logits)

            for i, (label, gt_label) in enumerate(zip(labels, gt_labels)):
                if label == gt_label:
                    num_cl += 1
                    if pred[i] == label.data: cl_c += 1
                    else: cl_w += 1
                else:
                    num_nl += 1
                    if pred[i] == gt_label.data: nl_c += 1
                    elif pred[i] == label.data: nl_m += 1
                    else: nl_w += 1

    acc = (cl_c + nl_c) / (num_cl + num_nl)
    try:
        cl_c /= num_cl
        cl_w /= num_cl
    except:
        cl_c = cl_w = 0
    try:
        nl_c /= num_nl
        nl_m /= num_nl
        nl_w /= num_nl
    except:
        nl_c = nl_m = nl_w = 0
    
    return acc, cl_c, cl_w, nl_c, nl_m, nl_w

def evaluate_accuracy(model, dataset_size, client_loader, args):
    accuracy = np.zeros(args.num_classes)
    true_labels = []
    pred_labels = []
    confidences = []
    
    model.eval()
    for i, (indices, inputs, labels, gt_labels) in enumerate(client_loader):
        true_labels += labels.tolist()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            conf, pred = torch.max(F.softmax(logits, dim=1), 1)
            pred_labels += pred.tolist()
            confidences += conf.tolist()
            correct = (pred == labels.data)
            for c in labels[correct]:
                accuracy[c] += 1.0

    accuracy = (accuracy / (dataset_size / args.num_classes)) * 100
    acc = round((np.sum(accuracy) / args.num_classes), 2)
    std = round(np.std(accuracy), 4)
    min = round(np.min(accuracy), 2)
    max = round(np.max(accuracy), 2)
    
    ece = []
    fig = None
    if args.calibration_check:
        ece, fig = reliability_diagram(np.array(true_labels), np.array(pred_labels), np.array(confidences))
        
    return acc, std, min, max, ece, fig

CRITERION = {'ce': cross_entropy, 'elr': elr_loss, 'elr+distil': distil_elr_loss}

def personalized_accuracy(model, server_weight, dataset_sizes, client_loader, class_num, args):
    client_weight = {}
    client_momentum = {}
    for client in range(args.num_clients):
        client_weight[client] = copy.deepcopy(server_weight)
        client_momentum[client] = {}
    
    if 'elr' in args.local_criterion:
        criterion = CRITERION[args.local_criterion](sum(dataset_sizes['train'].values()), class_num, args)
    else:
        criterion = CRITERION[args.local_criterion]
    
    client_weight, _, _ = client_opt(model, client_loader['train'], dataset_sizes['train'], server_weight, client_weight, client_momentum, criterion, args, args.num_rounds, validation=True)
    
    accs = []
    for client in range(args.num_clients):
        client_weight[client] = cpu_to_gpu(client_weight[client], args.device)
        if args.memorization_check:
            accs += memorization_check(model, client_weight[client], dataset_sizes['valid'][client], client_loader['valid'][client], args)
        else:
            model.load_state_dict(client_weight[client])
            model.eval()
            acc = 0
            for i, (indices, inputs, labels, gt_labels) in enumerate(client_loader['valid'][client]):
                inputs = inputs.to(args.device)
                gt_labels = gt_labels.to(args.device)

                with torch.set_grad_enabled(False):
                    logits = model(inputs)
                    pred = prediction(logits)

                    correct = (pred == gt_labels)
                    acc += correct.float().sum().item()
            acc = acc / dataset_sizes['valid'][client] * 100
            accs.append(acc)

    return accs