import argparse
import os
from utils.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser(description='FedAdaSparse Unified Framework')

    parser.add_argument('--method', type=str, default='fedadasparse',
                        choices=['fedadasparse'],
                        help='Federated method to run')
    parser.add_argument('--data_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'fashionmnist', 'svhn', 'tiny_imagenet'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_online_clients', type=int, default=10)
    parser.add_argument('--non_iid_alpha', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if True else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--save_path', type=str, default=os.path.join('./', 'result/'))

    parser.add_argument('--num_rounds', type=int, default=8)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--tau_percentile', type=float, default=20)
    parser.add_argument('--pilot_epochs', type=int, default=8)
    parser.add_argument('--epochs_per_layer', type=int, default=10)
    parser.add_argument('--min_active_layers', type=int, default=6)
    parser.add_argument('--max_active_layers', type=int, default=16)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr', type=float, default=5e-3)

    parser.add_argument('--aggregation', type=str, default='weighted',
                        choices=['equal', 'weighted'])

    parser.add_argument('--freeze_bn', action='store_true', default=False)
    parser.add_argument('--save_communication', action='store_true', default=True)

    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True

    return args
