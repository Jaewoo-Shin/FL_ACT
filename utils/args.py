import argparse

__all__ = ['parse_args', 'align_args']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

        
def parse_args():
    parser = argparse.ArgumentParser()

    ############ experimental settings ###############
    parser.add_argument('--device', 
                    help='gpu device;', 
                    type=str,
                    default='cuda:0')
    parser.add_argument('--seed',
                    help='seed for experiment',
                    type=int,
                    default=0)
    
    ######### record performance per epochs #########
    parser.add_argument('--client_accuracy',
                        help='print client accuracies for each epoch',
                        default=True,
                        action='store_true')
    parser.add_argument('--memorization_check',
                        help='print the measures that can show how much memorization occurs',
                        default=False,
                        action='store_true')
    parser.add_argument('--calibration_check',
                        help='print ECE for each epoch and draw reliability diagram and',
                        default=False,
                        action='store_true')

    #################### dataset ####################
    # basic settings
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    required=True,
                    choices=['dirichlet-mnist', 'dirichlet-cifar10', 'dirichlet-cifar100', 'dirichlet-fmnist', 'emnist', 'fed-cifar100', 'synthetic', 'landmark-g23k', 'landmark-g160k'])
    parser.add_argument('--data-dir', 
                    help='dir for dataset;',
                    type=str,
                    default='./data')
    parser.add_argument('--batch-size',
                    help='batch size of local data on each client;',
                    type=int,
                    default=128)
    parser.add_argument('--pin-memory', 
                    help='argument of pin memory on DataLoader;',
                    action='store_true')
    parser.add_argument('--num-workers', 
                    help='argument of num workers on DataLoader;',
                    type=int,
                    default=0)
    parser.add_argument('--num-clients', 
                    help='number of clients;',
                    type=int,
                    default=20)
    
    # non-iidness
    parser.add_argument('--dist-mode', 
                    help='which criterion to use on distributing the dataset;',
                    type=str,
                    default='class',
                    choices=['class', 'client'])
    parser.add_argument('--non-iid', 
                    help='dirichlet parameter to control non-iidness of dataset;',
                    type=float,
                    default=100.0)
    
    # for input regularization
    parser.add_argument('--input-reg',
                       help='whether cutmix or mixup',
                       type=str,
                       default=None)
    parser.add_argument('--cutmix',
                    help='probability of cutmix in local training;',
                    type=float,
                    default=0.0)
    
    # for noisy label
    parser.add_argument('--noise',
                       help='noise for dataset',
                       type=float,
                       default=0.0)
    parser.add_argument('--noise-type',
                       help='noise type for generating dataset',
                       type=str,
                       default='sym')    
    parser.add_argument('--noise-momentum',
                       help='momentum update for generated dataset',
                       type=float,
                       default=0.9)  
    
    ##################### model #####################
    
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    required=True,
                    choices=['lenet', 'lenetcontainer', 'vgg11', 'vgg11-bn', 'vgg13', 'vgg13-bn', 'vgg16', 'vgg16-bn', 'vgg19', 'vgg19-bn', 'resnet8', 'resnet18', 'resnet34', 'resnet50', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'simpleNet3', 'simpleNet4', 'simpleNet5', 'simpleNet6', 'simpleNet7', 'simpleNet4_bn', 'simpleNet4_sc'])
    # example : --model-kwargs num_classes=10
    parser.add_argument('--model-kwargs',
                        dest='model_kwargs',
                        action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    
    
    ################## server_opt ###################
    parser.add_argument('--algorithm',
                    help='which algorithm to select clients;',
                    type=str,
                    default='fedavg')
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=100)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=str,
                    default='20')
    parser.add_argument('--global-momentum',
                    help='whether to use momentum in server optimizers;',
                    type=float,
                    default=0.0)
    
    # client participation scheduler
    parser.add_argument('--cl-scheduler',
                       help = 'client participation scheduler',
                       type = str,
                       default='uniform',
                       choices=['multistep', 'exponential', 'cosine', 'uniform', 'linear'])
    parser.add_argument('--cl-decay',
                       help = 'decay for client participation multistep/exponential scheduler',
                       type = float,
                       default = 0.)
    parser.add_argument('--cl-decay-per-round',
                       help = 'decay round for client participation scheduler',
                       type = int,
                       default = 0)
    parser.add_argument('--cl-milestones',
                       help = 'milestones for client participation scheduler',
                       type=str,
                       default='0')
    parser.add_argument('--cl_period',
                       help = 'period for client participation cosine scheduler',
                       type=int,
                       default=10)
    parser.add_argument('--cl_par_min',
                       help = 'minimum number of participating clients in cosine scheduler',
                       type=int,
                       default=5)
    
    ################## client_opt ###################
    parser.add_argument('--num-epochs',
                    help='number of rounds to local update;',
                    type=int,
                    default=1)
    # criterion
    parser.add_argument('--local-criterion',
                    help='criterion to use in local training;',
                    type=str,
                    default='ce')
    parser.add_argument('--elr_lambda', type=float, default=3.0)
    parser.add_argument('--elr_beta', type=float, default=0.7)
    parser.add_argument('--elr_clamp', type=float, default=1e-4)
    parser.add_argument('--elr_init', type=str, choices=['zero', 'uniform'], default='zero')
    parser.add_argument('--smoothing',
                    help='smoothing epsilon for label smoothing method',
                    type=float,
                    default=0.0)
    parser.add_argument('--temperature',
                    help='temperature value for logit value',
                    type=float,
                    default=1.0)
    parser.add_argument('--tcp-aware',
                    help='new loss function that aware the tcp value',
                    action='store_true')
    
    # optimizer
    parser.add_argument('--local-optimizer',
                    help='optimizer to use in local training;',
                    type=str,
                    default='sgd')
    parser.add_argument('--local-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=0.1)
    parser.add_argument('--wd',
                    help='weight decay lambda hyperparameter in local optimizers;',
                    type=float,
                    default=1e-4)
    parser.add_argument('--mu',
                    help='fedprox mu hyperparameter in local optimizers;',
                    type=float,
                    default=0.0)
    parser.add_argument('--nesterov',
                    help='nesterov switch for local momentum',
                    action='store_true')
    parser.add_argument('--local-momentum',
                    help='whether to use momentum in local optimizers;',
                    type=float,
                    default=0.9)
    parser.add_argument('--lr-decay',
                    help = 'learning rate decay',
                    type = float,
                    default = 0.1)
    parser.add_argument('--milestones',
                    help= 'milestones for step scheduler',
                    type = str,
                    default = '50,75')
    parser.add_argument('--sch-type',
                       help = 'scheduler for local client learning rate',
                       type=str,
                       default='uniform')
    
    # mab settings
    parser.add_argument('--reward',
                    help='which reward function to use;',
                    type=str,
                    choices=['singular', 'val-loss', 'val-acc', 'bias'],
                    default=None)
    parser.add_argument('--combination-bandit',
                    help='which approach for arms of bandit',
                    action='store_true')
    parser.add_argument('--prior-params',
                    help='whether to use prior knowledge to init mab params;',
                    action='store_true')
    parser.add_argument('--mab-decay',
                    help='how much to decay the mab params;',
                    type=float,
                    default=1.0)
    
    # knowledge distillation
    parser.add_argument('--logit-distillation',
                       help='regularization of proximal term in logit level',
                       default=None,
                       type=str)
    parser.add_argument('--teacher',
                       help='self-distillation teacher',
                       default=None,
                       type=str)
    parser.add_argument('--retrain',
                       help='reinitialize',
                       default=None,
                       type=str)
    
    ############### deleted argument ################
#     sub model use
#     parser.add_argument('--sub-model',
#                     help='name of sub_model;',
#                     type=str,
#                     default='')
#     parser.add_argument('--sub-model-kwargs',
#                         dest='sub_model_kwargs',
#                         action=StoreDictKeyPair,
#                         metavar="KEY1=VAL1,KEY2=VAL2...")
#     parser.add_argument('--sub-model-num-epochs', 
#                     help='num epohcs for sub_model training;',
#                     type=int,
#                     default=100)
#     aggregation agnostic_fl
#     parser.add_argument('--agnostic-fl',
#                     help='train an aggregate weights, refers to agnostic fl;',
#                     type=str2bool,
#                     default=False)
    ################################################
    
    return parser.parse_args()


def align_args(args):
    # align other argument according to args.selection
    if args.algorithm == 'fedavg' or args.algorithm == 'fedavg-dropout':
        args.mab_decay = 1.0
        args.mu = 0.0
        args.full_part = True
        args.combination_bandit = False
        
    elif args.algorithm == 'fedavg-conv' or args.algorithm == 'fedavg-dropout-conv':
        args.mab_decay = 1.0
        args.mu = 0.0
        args.full_part = False
        args.combination_bandit = False
        
    elif args.algorithm == 'fedprox':
        assert args.mu != 0.0
        args.full_part = False
        args.combination_bandit = False
        
    elif args.algorithm == 'mab-ucb':
        args.mab_decay = 1.0
        args.mu = 0.0
        args.full_part = False
        args.use_comb = True
        
    elif args.algorithm == 'mab-thompson':
        args.mu = 0.0
        args.full_part = False
        args.use_comb = True
    
    else:
        raise NotImplemented
        
    if args.cl_scheduler == 'exponential':
        args.clients_per_round = '22'
        args.cl_decay = 0.95
        args.cl_decay_per_round = 3
        
    elif args.cl_scheduler == 'linear':
        args.clients_per_round = '22'
        args.cl_decay = 2
        args.cl_decay_per_round = 11
        
    elif args.cl_scheduler == 'multistep':
        args.clients_per_round = '15'
        args.cl_decay = 0.5
        args.cl_milestones = '50,75'
        
    elif args.cl_scheduler == 'cosine':
        args.clients_per_round = '15'
        args.cl_period = 10
        args.cl_par_min = 5
    else:
        pass
        
    return args
