import os
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['data_plotter', 'clients_acc_plotter', 'train_test_acc_plotter', 'selected_clients_plotter', 'test_acc_plotter', 'mab_plotter']


def data_plotter(data_dist, img_filename):
    num_clients = len(data_dist)
    num_classes = len(data_dist[0])
    tmp = img_filename.split('_')
    non_iid = float(tmp[3])
    valid_size = int(tmp[4])
    
    classwise_data_stat = {}
    for cls in range(num_classes):
        classwise_data_stat[cls] = [data_dist[c][cls] for c in range(num_clients)]
        
    x = range(num_clients)
    labels = ['C{}'.format(number) for number in range(num_clients)]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    tmp = np.zeros(num_clients)

    f = plt.figure(figsize=(20,10))
    for i in range(num_classes):
        plt.bar(x, classwise_data_stat[i], bottom = tmp, label = classes[i])
        bottom = np.add(classwise_data_stat[i], tmp)
        tmp += classwise_data_stat[i]
    plt.rc('legend', fontsize=18)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if 'class' in img_filename:
        num = int((50000 - valid_size * num_classes) / num_clients)
        ax.set_ylim(0, num * 1.05)
        
    #plt.title('Dirichilet data distribution with non-iid {}'.format(non_iid))
    #plt.xlabel('Clients')
    #plt.ylabel('number of data')
    plt.legend(loc='upper center', ncol=10, bbox_to_anchor= (0.5,-0.05))
    
    f.savefig(img_filename + '/data_dist.png', bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    
def __get_clients_acc(model, client_loader, num_classes, num_clients, device):
    prediction = lambda logits : torch.argmax(logits, 1)
    
    clients_acc = {}
    for client in range(num_clients):
        clients_acc[client] = {}
        for i in range(num_classes):
            clients_acc[client][i] = 0
        
        total = 0
        model.eval()
        for i, data in enumerate(client_loader['train'][client]):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            total += data[1].shape[0]
            with torch.set_grad_enabled(False):
                logits = model(inputs)
                pred = prediction(logits)
                correct = (pred == labels.data)
                for cls in labels[correct]:
                    clients_acc[client][cls.item()] += 1.0
        
        clients_acc[client]['cr'] = np.sum([clients_acc[client][num] for num in range(10)])
        clients_acc[client]['total'] = total
    
    classwise_acc_stat = {}
    for num in range(num_classes):
        for key in clients_acc.keys():
            classwise_acc_stat[num] = round(clients_acc[key][num] / (clients_acc[key]['cr'] + 1e-10), 2)
        
    return clients_acc, classwise_acc_stat
                
    
def __get_train_test_acc(model, client_loader, num_classes, num_clients, device):
    prediction = lambda logits : torch.argmax(logits, 1)
    
    train_test_acc = {'train': {}, 'test': {}}
    for key in train_test_acc.keys():
        for i in range(num_classes):
            train_test_acc[key][i] = 0.
            
    total = 0
    for client in range(num_clients):
        for i, data in enumerate(client_loader['train'][client]):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            total += data[1].shape[0]
            with torch.set_grad_enabled(False):
                logits = model(inputs)
                pred = prediction(logits)
                correct = (pred == labels.data)
                for cls in labels[correct]:
                    train_test_acc['train'][cls.item()] += 1.0
                    
    train_test_acc['train']['cr'] = np.sum([train_test_acc['train'][num] for num in range(num_classes)])
    train_test_acc['train']['total'] = total
    
    total = 0                
    for i, data in enumerate(client_loader['test']):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            total += data[1].shape[0]
            with torch.set_grad_enabled(False):
                logits = model(inputs)
                pred = prediction(logits)
                correct = (pred == labels.data)
                for cls in labels[correct]:
                    train_test_acc['test'][cls.item()] += 1.0
        
    train_test_acc['test']['cr'] = np.sum([train_test_acc['test'][num] for num in range(num_classes)])
    train_test_acc['test']['total'] = total
        
    return train_test_acc


def clients_acc_plotter(model, img_filename, client_loader, args, rounds):
    num_classes = args.num_classes
    num_clients = args.num_clients
    device = args.device
    non_iid = float(img_filename.split('_')[3])
    local_epoch = float(img_filename.split('_')[5])
    
    clients_acc, classwise_acc_stat= __get_clients_acc(model, client_loader, num_classes, num_clients, device)
    
    x = range(num_clients)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels = ['C{}'.format(number) for number in range(num_clients)]
    tmp = np.zeros(num_clients)

    f = plt.figure(figsize=(20,10))
    for i in range(num_classes):
        plt.bar(x, np.multiply(classwise_acc_stat[i], [clients_acc[key]['cr'] / clients_acc[key]['total'] for key in clients_acc]), bottom = tmp, label = classes[i])
        
        tmp = np.add(np.multiply(classwise_acc_stat[i], [clients_acc[key]['cr'] / clients_acc[key]['total'] for key in clients_acc]), tmp)
    
    plt.rc('legend', fontsize=18)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    
    ax = plt.subplot()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #plt.title('Dirichilet data distribution with non-iid {}, local_epoch {}'.format(non_iid, local_epoch))
    #plt.xlabel('Clients')
    ax.set_ylim([0, 1.05])
    #plt.ylabel('Accuracy')
    plt.legend(loc='upper center', ncol=10, bbox_to_anchor= (0.5,-0.05))

    f.savefig(img_filename + '/clients_acc_[round_%d].png' % rounds, bbox_inches = 'tight')
    plt.close()
    
    
def train_test_acc_plotter(model, img_filename, client_loader, args, rounds):
    num_classes = args.num_classes
    num_clients = args.num_clients
    device = args.device
    non_iid = float(img_filename.split('_')[3])
    valid_size = int(img_filename.split('_')[4])
    local_epoch = float(img_filename.split('_')[5])
    
    train_test_acc = __get_train_test_acc(model, client_loader, num_classes, num_clients, device)
        
    x = range(num_classes)
    labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    f = plt.figure(figsize=(20,10))
    n = 1  # This is our first dataset (out of 2)
    t = 2 # Number of dataset
    d = 10 # Number of sets of bars
    w = 0.8 # Width of each bar
    store1_x = [t*element + w*n for element in range(d)]

    plt.bar(store1_x, [train_test_acc['train'][key] / (5000 - valid_size) for key in x], label = 'train')

    n = 2  # This is our second dataset (out of 2)
    t = 2 # Number of dataset
    d = 10 # Number of sets of bars
    w = 0.8 # Width of each bar
    store2_x = [t*element + w*n for element in range(d)]
    plt.bar(store2_x, [train_test_acc['test'][key] / 1000 for key in x], label = 'test')
    
    plt.rc('legend', fontsize=18)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    #plt.title('Dirichilet data distribution with non-iid {}, local_epoch {}'.format(non_iid, local_epoch))
    #plt.xlabel('Clients')
    ax.set_xticks(np.add(store2_x, store1_x) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.05])
    #plt.ylabel('Accuracy')
    plt.legend()
    
    f.savefig(img_filename + '/train_test_acc_[round_%d].png' % rounds, bbox_inches = 'tight')
    plt.close()


def selected_clients_plotter(selected_clients_num, img_filename, args):
    num_clients = args.num_clients
    num_classes = args.num_classes
    
    x = range(num_clients)
    labels = ['C{}'.format(number) for number in range(num_clients)]

    f = plt.figure(figsize=(20,10))
    plt.bar(x, selected_clients_num)
    plt.axhline(args.num_rounds, x[0], x[-1], color='red')
    
    plt.rc('legend', fontsize=18)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #plt.title('Dirichilet data distribution with non-iid {}'.format(non_iid))
    #plt.xlabel('Clients')
    #plt.ylabel('number of data')
    plt.legend(loc='upper center', ncol=10, bbox_to_anchor= (0.5,-0.05))
    
    f.savefig(img_filename + '/selected_clients_num.png', bbox_inches = 'tight')
    plt.close()


def test_acc_plotter(test_acc, img_filename, args):    
    x = range(args.num_rounds)

    f = plt.figure(figsize=(20,10))
    plt.plot(x, test_acc)
    
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    ax.set_xticks(x)
    #plt.title('Dirichilet data distribution with non-iid {}'.format(non_iid))
    #plt.xlabel('Clients')
    #plt.ylabel('number of data')
    
    f.savefig(img_filename + '/test_acc.png', bbox_inches = 'tight')
    plt.close()


def mab_plotter(mab_params, r, img_filename, args):
    assert 'thompson' in args.algorithm
    
    f = plt.figure(figsize=(20,10))
    for arm in mab_params['alpha'].keys():
        a = mab_params['alpha'][arm]
        b = mab_params['beta'][arm]
        x = np.linspace(-1, 1, 100)
        y = beta.pdf(x,a,b)#, scale=100, loc=-50)
        plt.plot(x,y, labels='client %s' % arm)
    plt.legend()
        
    f.savefig(img_filename + '/mab_params_round_%s.png', bbox_inches = 'tight')
    plt.close()