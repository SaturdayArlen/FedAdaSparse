import torch
import torch.nn as nn
import torch.optim as optim
from utils import compute_accuracy, train_one_epoch
from .mem_utils import (get_client_params, set_client_params,
                        aggregate_layer_wise, aggregate_layer_wise_weighted,
                        get_client_communication_size)


def federated_train_fedadasparse(client_models, client_queues, client_activated,
                                 client_train_loaders, client_test_loaders,
                                 val_loader, global_params, rounds=8,
                                 local_epochs=2, lr=1e-3,
                                 aggregation='weighted', decay=0.97,
                                 freeze_bn=False, weight_decay=1e-5,
                                 device='cuda'):
    criterion = nn.CrossEntropyLoss()
    history = {
        'rounds': [],
        'avg_personalized_acc': [],
        'std_personalized_acc': [],
        'global_acc': [],
        'comm_per_round': [],
    }
    num_clients = len(client_models)
    client_sizes = [len(loader.dataset) for loader in client_train_loaders]
    total_comm = 0

    if freeze_bn:
        for model in client_models:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    for round_idx in range(rounds):
        print(f"\n===== 联邦轮次 {round_idx+1}/{rounds} =====")
        client_updates = []
        round_comm = 0
        current_lr = lr * (decay ** round_idx)

        for client_id in range(num_clients):
            model = client_models[client_id]
            queue = client_queues[client_id]
            activated = client_activated[client_id]
            train_loader = client_train_loaders[client_id]
            model.to(device)

            set_client_params(model, queue, activated, global_params)
            round_comm += get_client_communication_size(queue, activated)

            if len(activated) == 0:
                client_updates.append({
                    'activated': activated,
                    'params': {},
                    'num_samples': client_sizes[client_id]
                })
                continue

            trainable_params = []
            for idx in activated:
                lora_module = queue[idx]
                trainable_params.extend([lora_module.encoder.weight,
                                         lora_module.decoder.weight])
            optimizer = optim.Adam(trainable_params, lr=current_lr,
                                   weight_decay=weight_decay)
            model.train()
            for epoch in range(local_epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device)

            params = get_client_params(model, queue, activated)
            client_updates.append({
                'activated': activated,
                'params': params,
                'num_samples': client_sizes[client_id]
            })
            round_comm += get_client_communication_size(queue, activated)

        if aggregation == 'weighted':
            global_params = aggregate_layer_wise_weighted(client_updates)
        else:
            global_params = aggregate_layer_wise(client_updates)

        total_comm += round_comm
        history['comm_per_round'].append(round_comm)

        personalized_accs = []
        for client_id in range(num_clients):
            model = client_models[client_id]
            test_loader = client_test_loaders[client_id]
            acc = compute_accuracy(model, test_loader, device)
            personalized_accs.append(acc)
        avg_acc = sum(personalized_accs) / num_clients
        std_acc = torch.std(torch.tensor(personalized_accs)).item()

        eval_model = client_models[0]
        eval_queue = client_queues[0]
        set_client_params(eval_model, eval_queue, client_activated[0], global_params)
        global_acc = compute_accuracy(eval_model, val_loader, device)

        history['rounds'].append(round_idx + 1)
        history['avg_personalized_acc'].append(avg_acc)
        history['std_personalized_acc'].append(std_acc)
        history['global_acc'].append(global_acc)

        print(f"轮次 {round_idx+1}: 平均个性化精度 = {avg_acc:.2f}% ± {std_acc:.2f}%, "
              f"全局精度 = {global_acc:.2f}%, 本轮通信 = {round_comm/1024:.2f} KB")

    history['total_communication'] = total_comm
    return global_params, history
