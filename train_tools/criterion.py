import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.device import *
from models import *

import numpy as np
import copy

from .optimizer import *

__all__ = ['cross_entropy', 'elr_loss', 'distil_elr_loss', 'input_reg', 'mse_logit', 'kd_loss', 'gradientNoiseImage']

MODEL = {'lenet': LeNet, 'lenetcontainer': LeNetContainer, 'vgg11': vgg11,
         'vgg11-bn': vgg11_bn, 'vgg13': vgg13, 'vgg13-bn': vgg13_bn,
         'vgg16': vgg16, 'vgg16-bn': vgg16_bn, 'vgg19': vgg19, 'vgg19-bn': vgg19_bn,
         'resnet8': resnet8}

def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w = np.int(W*cut_rat)
    cut_h = np.int(H*cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def input_reg(inputs, labels, device, alpha=1.0, mode=None):

    assert mode not in ['cutmix'], 'Still in the process of develop.'

    if mode == 'mixup':
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(inputs.size()[0]).to(device)
        labels1 = labels.to(device)
        labels2 = labels[rand_index].to(device)
        inputs2 = copy.deepcopy(inputs)

        inputs = torch.autograd.Variable(lam * inputs + (1-lam) * inputs[rand_index,:,:,:]).to(device)

        return lam, inputs, labels1, labels2
    
    elif mode == 'cutmix':
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(inputs.size()[0]).to(device)
        labels1 = labels.to(device)
        labels2 = labels.to(device)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)
        inputs[:,:,bbx1:bbx2, bby1:bby2]=inputs[rand_index,:,bbx1:bbx2,bby1:bby2]
        lam = 1 - ((bbx2-bbx1) * (bby2-bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        
        inputs = torch.autograd.Variable(inputs).to(device)
        
        return lam, inputs, labels1, labels2
        
    else:
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels


def cross_entropy(output, label, classes=None, smoothing=0.0, temperature=1.0):
    assert 0.0 <= smoothing and smoothing <= 1.0, 'smoothing epsilon should be in the range between 0.0 and 1.0.'
    
    if smoothing != 0.0:
        criterion = LabelSmoothingLoss(classes, smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
            
    return (temperature ** 2) * criterion(output / temperature, label)

class elr_loss(nn.Module):
    '''
    https://github.com/shengliu66/ELR/blob/master/ELR/model/loss.py
    '''
    def __init__(self, dataset_size, classes, args):
        super(elr_loss, self).__init__()
        if args.elr_init == 'uniform':
            self.target = torch.full((dataset_size, classes), 1 / classes).to(args.device)
        else:
            self.target = torch.zeros(dataset_size, classes).to(args.device)
        self.classes = classes
        self.device = args.device
        self.lamb = args.elr_lambda
        self.beta = args.elr_beta
        self.clamp = args.elr_clamp
        self.smoothing = args.smoothing
        self.temperature = args.temperature

    def forward(self, output, label, index):
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, self.clamp, 1.0 - self.clamp)
        pred_ = pred.detach()
        pred_ /= pred_.sum(dim=1, keepdim=True)

        #Target Estimation
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * pred_

        loss_ce = cross_entropy(output, label, self.classes, self.smoothing, self.temperature)
        elr_reg = (1 - (self.target[index] * pred).sum(dim=1)).log().mean()
        loss_elr = loss_ce +  self.lamb * elr_reg
        return loss_elr

class distil_elr_loss(nn.Module):
    def __init__(self, dataset_size, classes, args):
        super(distil_elr_loss, self).__init__()
        self.device = args.device
        self.lamb = args.elr_lambda
        self.beta = args.elr_beta
        self.clamp = args.elr_clamp

    def forward(self, output, output_s, label):
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, self.clamp, 1.0 - self.clamp)
        pred_ = pred.detach()
        pred_ /= pred_.sum(dim=1, keepdim=True)
        
        pred_s = F.softmax(output_s, dim=1)
        pred_s = torch.clamp(pred_s, self.clamp, 1.0 - self.clamp)
        pred_s_ = pred_s.data.detach()
        pred_s_ /= pred_s_.sum(dim=1, keepdim=True)

        #Target Estimation
        target = self.beta * pred_s_ + (1 - self.beta) * pred_

        loss_ce = F.cross_entropy(output, label)
        elr_reg = (1 - (target * pred).sum(dim=1)).log().mean()
        loss_elr = loss_ce +  self.lamb * elr_reg
        return loss_elr

class LabelSmoothingLoss(nn.Module):
    '''
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    '''
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, targets):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
    
class TCPLoss(nn.Module):
    def __init__(self):
        super(TCPLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred, targets):
        x = pred.softmax(dim=-1)
        x = torch.index_select(x, 1, labels).diag()
        
#         loss = ((1.5 - x) * self.criterion(pred, targets)).mean()
        loss = ((2 - 2*x) * self.criterion(pred, targets)).mean()
        
        return loss
    
    
def mse_logit(output, teacher_output):
    
    return nn.MSELoss()(output, teacher_output)


def kd_loss(output, teacher_output, alpha=1.0, temperature=1.0):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
        
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    
    """
    kd_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1),
                                             F.softmax(teacher_output/T, dim=1)).type(torch.FloatTensor).cuda(gpu)
    
    kd_loss = kd_filter * torch.sum(kd_loss, dim=1) # kd filter is filled with 0 and 1.
    kd_loss = torch.sum(kd_loss) / torch.sum(kd_filter) * (alpha * T * T)
    """
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
                                                  F.softmax(teacher_output/T, dim=1)) * (alpha * max(T, T*T))

    
    return kd_loss


class AdaptiveLabelLoss(nn.Module):
    def __init__(self, device, classes, model, smoothing=0.0, dim=-1):
        super(AdaptiveLabelLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.device = device
        self.model = model
        self.makeConfusion()
        
    def makeConfusion(self, T=1):
        weight_matrix = list(self.model.parameters())[-2].data
        num = weight_matrix.shape[0]
        cos_si = nn.CosineSimilarity(dim=0)
        similarity = torch.zeros([num,num]).to(self.device)
        for i in range(num):
            for j in range(num):
                similarity[i,j] = cos_si(weight_matrix[i], weight_matrix[j]) * T
            #similarity[i,i] = 0.
            similarity[i] = torch.exp(similarity[i]) / (torch.sum(torch.exp(similarity[i])) - torch.exp(similarity[i,i]))
            similarity[i,i] = 0.
        self.confusion = similarity
        

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            for i in range(target.shape[0]):
                diri = torch.distributions.dirichlet.Dirichlet(torch.tensor(self.confusion[target[i]]))
                tmp = diri.sample()
                true_dist[i] = tmp * self.smoothing
                true_dist[i][target[i]] = self.confidence
                
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class GradientNoiseImage(nn.Module):
    def __init__(self, model, device):
        super(GradientNoiseImage, self).__init__()
        self.model = model
        self.device = device
        self.epsilon = 0.01
        self.freeze()
        
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def generate(self, size, clsidx, iters = 50, epsilon=0.15, clip_min=0., clip_max=1.):
        NoiseImage = Variable(torch.randn(size, 3, 32, 32)).to(self.device)
        NoiseImage.requires_grad=True
        label = (clsidx * torch.ones(size).long()).to(self.device)
        
        self.model.eval()
        optimizer = torch.optim.Adam([NoiseImage], self.epsilon)
        for _ in range(iters):
            NoiseImage.grad = torch.zeros_like(NoiseImage).to(self.device)
            NoiseImage.grad.detach_()
            
#             with torch.set_grad_enabled(True):
            loss = AdaptiveLabelLoss(self.device, 10, self.model, 0.3)(self.model(NoiseImage), label)
#                 loss = nn.CrossEntropyLoss()(self.model(NoiseImage), label)
            loss.backward()
#             NoiseImage = NoiseImage - self.epsilon * NoiseImage.grad
            optimizer.step()
#             d = epsilon * NoiseImage.grad.data.sign() # / torch.norm(X.grad.data.sign().view(size,-1), 1)

#             NoiseImage = NoiseImage - d

            if clip_max ==None and clip_min == None:
                clip_max = np.inf
                clip_min = -np.inf
            NoiseImage = Variable(torch.clamp(NoiseImage, clip_min, clip_max)).to(self.device)
            NoiseImage.requires_grad = True
        del loss
        
        return NoiseImage.detach_(), label
    
    
    
def gradientNoiseImage(weight, valid_size, device, args, prev_valid=None):
    
    weight = cpu_to_gpu(weight, device)
    model = MODEL[args.model](num_classes=args.num_classes, **args.model_kwargs) if args.model_kwargs else MODEL[args.model](num_classes=args.num_classes)
    model = model.to(device)
    model.load_state_dict(weight)
    generator = GradientNoiseImage(model, device)
    weight = gpu_to_cpu(weight)
    
    for clsidx in range(10):
        if clsidx == 0:
            dataloader, label = generator.generate(size=valid_size, clsidx=clsidx)
        else:
            tmploader, tmplabel = generator.generate(size=valid_size, clsidx=clsidx)
            dataloader = torch.cat([dataloader, tmploader], dim=0)
            label = torch.cat([label, tmplabel], dim=0)
            
    del generator
    
    if args.noise_momentum != 0.0 and prev_valid is not None:
        dataloader = dataloader * (1 - args.noise_momentum) + prev_valid * args.noise_momentum
    
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataloader, label), batch_size=500)