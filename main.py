import importlib
import torch
import numpy as np
import random

from option import args_parser
from model.models import create_resnet18
from dataset.data_utils import get_cifar_loaders
from dataset.sampling import split_non_iid

# 方法注册表: method -> model/ 下的包目录
METHOD_PACKAGES = {
    'fedadasparse': 'model.Ours_FedAdaSparse',        # Ours
}


def import_method_module(method, module_name):
    """按需导入 model/<方法文件夹>/<module_name>.py 模块"""
    return importlib.import_module(f'{METHOD_PACKAGES[method]}.{module_name}')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_fedadasparse(args):
    client_mod = import_method_module('fedadasparse', 'client')
    server_mod = import_method_module('fedadasparse', 'server')
    mem_mod = import_method_module('fedadasparse', 'mem_utils')
    client_adaptive_activation = client_mod.client_adaptive_activation
    federated_train_fedadasparse = server_mod.federated_train_fedadasparse
    get_client_params = mem_mod.get_client_params
    aggregate_layer_wise = mem_mod.aggregate_layer_wise

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"方法: FedAdaSparse")
    print(f"数据集: {args.data_name.upper()}")

    _, global_val_loader, trainset, _, num_classes = get_cifar_loaders(
        args.data_name, batch_size=args.batch_size
    )
    client_train_loaders, client_val_loaders = split_non_iid(
        trainset, args.num_clients, alpha=args.non_iid_alpha,
        num_classes=num_classes, val_ratio=args.val_ratio
    )
    print(f"已划分 {args.num_clients} 个 Non-IID 客户端，"
          f"alpha={args.non_iid_alpha}，类别数={num_classes}")

    client_models = []
    client_queues = []
    client_activated = []

    print("=== 阶段一：各客户端本地自适应激活 ===")
    for cid in range(args.num_clients):
        print(f"\n--- 客户端 {cid} ---")
        model = create_resnet18(num_classes=num_classes, pretrained=True)
        model, queue, activated = client_adaptive_activation(
            model, client_train_loaders[cid], client_val_loaders[cid],
            device=device,
            lora_rank=args.lora_rank,
            tau_percentile=args.tau_percentile,
            pilot_epochs=args.pilot_epochs,
            epochs_per_layer=args.epochs_per_layer,
            pilot_lr=args.lr,
            activation_lr=args.lr,
            min_layers=args.min_active_layers,
            max_layers=args.max_active_layers,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr
        )
        client_models.append(model)
        client_queues.append(queue)
        client_activated.append(activated)
        print(f"客户端 {cid} 最终激活层: {activated}")

    print("\n=== 初始聚合（等权） ===")
    initial_updates = []
    for cid in range(args.num_clients):
        params = get_client_params(client_models[cid], client_queues[cid],
                                   client_activated[cid])
        initial_updates.append({
            'activated': client_activated[cid],
            'params': params
        })
    global_params = aggregate_layer_wise(initial_updates)

    print("\n=== 联邦训练 ===")
    final_params, history = federated_train_fedadasparse(
        client_models, client_queues, client_activated,
        client_train_loaders, client_val_loaders,
        global_val_loader, global_params,
        rounds=args.num_rounds, local_epochs=args.local_epochs,
        lr=args.lr, aggregation=args.aggregation,
        freeze_bn=args.freeze_bn, device=device
    )

    print("\n" + "=" * 60)
    print("FedAdaSparse 最终评估报告")
    print("=" * 60)
    print(f"数据集: {args.data_name.upper()}, 客户端: {args.num_clients}, "
          f"alpha={args.non_iid_alpha}")
    avg_active = sum(len(a) for a in client_activated) / args.num_clients
    min_active = min(len(a) for a in client_activated)
    max_active = max(len(a) for a in client_activated)
    print(f"激活层数分布: 均值={avg_active:.1f}, 最小={min_active}, 最大={max_active}")
    print(f"最终平均个性化精度: {history['avg_personalized_acc'][-1]:.2f}% ± "
          f"{history['std_personalized_acc'][-1]:.2f}%")
    print(f"最终全局精度: {history['global_acc'][-1]:.2f}%")
    print(f"总通信量: {history['total_communication']/1024/1024:.2f} MB")
    print(f"精度历史: {history['avg_personalized_acc']}")


def main():
    args = args_parser()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"FedAdaSparse Unified Framework")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.data_name}")
    print(f"Clients: {args.num_clients}, Non-IID alpha: {args.non_iid_alpha}")
    print(f"{'='*60}\n")

    if args.method == 'fedadasparse':
        run_fedadasparse(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == '__main__':
    main()
