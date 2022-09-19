import os
import numpy as np
import pandas as pd
import decimal

__all__ = ['make_log', 'modify_path']


def make_log(args):
    round_number_clients = round(float(eval(args.clients_per_round)))
    exp = '{}({})clients'.format(args.num_clients, round_number_clients)
    if 'elr' in args.local_criterion:
        exp += '_' + str(args.elr_lambda) + '_' + str(args.elr_beta) + '_' + '%.0e' % decimal.Decimal(str(args.elr_clamp))
    
    for tmp_path in ['./results', './model_checkpoints']:
        os.makedirs(os.path.join(tmp_path, args.dataset, args.model, exp), exist_ok=True)
    
    filename = args.dataset + '/' + args.model + '/' + exp + '/' \
                    + str(args.seed) + '-' + str(args.num_clients) + '_' + str(args.clients_per_round) \
                    + '_' + str(args.num_epochs) + '_' + str(args.num_rounds) + '_' + str(args.non_iid) + '_' + str(args.local_momentum) \
                    + '_' + str(args.local_optimizer) + '_' + str(args.local_lr) + '_' + str(args.lr_decay) + '_' + str(args.local_criterion) + '_' + str(args.model_kwargs) \
                    + '_' + str(args.wd)

    log_filename = './results/' + filename + '.csv'
    
    check_filename = './model_checkpoints/' + filename + '.t1'
    
    img_filename = './results/' + filename + '.png'
        
    log_columns = ['server_acc', 'server_std', 'server_min', 'server_max']
    if args.calibration_check:
        log_columns += ['server_ece']
    if args.memorization_check:
        log_columns += ['clean_correct', 'clean_wrong', 'noisy_correct', 'noisy_memorized', 'noisy_wrong']
    if args.client_accuracy:
        for client in range(args.num_clients):
            if args.memorization_check:
                for measure in ['acc', 'clean_correct', 'clean_wrong', 'noisy_correct', 'noisy_memorized', 'noisy_wrong']:
                    log_columns += [str(client) + '_' + measure]
            else:
                log_columns += [str(client)]

    log_pd = pd.DataFrame(np.zeros([args.num_rounds + 1, len(log_columns)]), columns = log_columns)

    return log_filename, log_pd, check_filename, img_filename


def modify_path(args):
    round_number_clients = round(float(eval(args.clients_per_round)))
    exp = '{}({})clients'.format(args.num_clients, round_number_clients)
    if 'elr' in args.local_criterion:
        exp += '_' + str(args.elr_lambda) + '_' + str(args.elr_beta) + '_' + '%.0e' % decimal.Decimal(str(args.elr_clamp))
    
    filename = args.dataset + '/' + args.model + '/' + exp + '/' \
                    + str(args.seed) + '_' + str(args.num_clients) + '_' + str(args.clients_per_round) \
                    + '_' + str(args.num_epochs) + '_' + str(args.num_rounds) + '_' + str(args.non_iid) + '_' + str(args.local_momentum) \
                    + '_' + str(args.local_optimizer) + '_' + str(args.local_lr) + '_' + str(args.lr_decay) + '_' + str(args.local_criterion) + '_' + str(args.model_kwargs) \
                    + '_' + str(args.wd)

    log_filename = './results/' + filename + '.csv'
    
    check_filename = './model_checkpoints/' + filename + '.t1'
    
    img_filename = './results/' + filename + '.png'
    
    return log_filename, check_filename, img_filename