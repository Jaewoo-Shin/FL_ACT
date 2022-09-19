#!/bin/bash

num_clients_list="100"
participate_client_ratio_list="*0.1 *0.2 *0.3 *0.4"
non_iid_list="0.1"
epoch_list="5"
seed_list="0"
logit_distillation="none"
num_filters="64"
lr_rates="0.005"
activations="none tanh simple_tanh relu leaky_relu"
num_aggregations="100"
for seed in $seed_list; do
    for num_clients in $num_clients_list; do
        for participate_client_ratio in $participate_client_ratio_list; do
            for non_iidness in $non_iid_list; do
                for epochs in $epoch_list; do
                    for filter_size in $num_filters; do
                        for lr_rate in $lr_rates; do
                            for activation in $activations; do
                                for aggregation_step in $num_aggregations; do
                                    python main_best.py --device 'cuda:0' --seed $seed \
                                                        --model "resnet34" \
                                                        --dataset "dirichlet-cifar10" --non-iid $non_iidness --noise 0.0 \
                                                        --num-rounds $aggregation_step --batch-size 64 \
                                                        --num-clients $num_clients --num-epochs $epochs \
                                                        --clients-per-round "$num_clients $participate_client_ratio" \
                                                        --algorithm 'fedavg' --dist-mode "class" \
                                                        --local-lr $lr_rate --lr-decay 0.1 --milestones "50,75" \
                                                        --local-optimizer "sgd" --local-momentum 0.9 --wd 1e-4 \
                                                        --model-kwargs activation=$activation
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done