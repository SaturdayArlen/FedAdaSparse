import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import compute_loss, train_one_epoch, compute_accuracy, warmup_classifier
from .models import build_conv_lora_queue, freeze_backbone


def auto_calibrate_threshold(model, queue, train_loader, val_loader,
                              p=50, pilot_epochs=4, lr=1e-3, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    contributions = []
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'encoder' in name or 'decoder' in name:
            param.requires_grad = True

    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()

    baseline_loss = compute_loss(model, val_loader, criterion, device)
    print(f"初始验证损失: {baseline_loss:.4f}")

    for layer_idx, lora in enumerate(queue):
        lora.activate()
        print(f"\n试点第 {layer_idx} 层（高层优先）...")
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

        for epoch in range(pilot_epochs):
            model.train()
            for m in bn_modules:
                m.eval()
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        current_loss = compute_loss(model, val_loader, criterion, device)
        contribution = baseline_loss - current_loss
        contributions.append(contribution)
        print(f"  贡献: {contribution:.6f}")

        baseline_loss = current_loss

    for lora in queue:
        lora.deactivate()

    if len(contributions) > 0:
        tau_raw = np.percentile(contributions, p)
        tau = tau_raw
        print(f"\n贡献序列: {contributions}")
        print(f"百分位数 p={p} 对应的原始阈值 = {tau_raw:.6f}, 最终 τ = {tau:.6f}")
    else:
        tau = 0.005

    for m in bn_modules:
        m.train()

    return tau, contributions


def early_stop_activation(model, queue, train_loader, val_loader,
                          tau=0.01, epochs_per_layer=3, lr=1e-3, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    activated = []

    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()

    model.to(device)
    baseline_loss = compute_loss(model, val_loader, criterion, device)
    print(f"初始验证损失: {baseline_loss:.4f}")

    for layer_idx, lora in enumerate(queue):
        lora.activate()
        print(f"\n--- 测试第 {layer_idx} 层（高层优先） ---")

        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)

        for epoch in range(epochs_per_layer):
            model.train()
            for m in bn_modules:
                m.eval()
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        current_loss = compute_loss(model, val_loader, criterion, device)
        contribution = baseline_loss - current_loss
        print(f"  验证损失: {current_loss:.4f}, 贡献: {contribution:.6f}")

        if contribution >= tau:
            activated.append(layer_idx)
            baseline_loss = current_loss
            print(f"  ✓ 接受该层，当前激活: {activated}")
        else:
            lora.deactivate()
            print(f"  ✗ 贡献不足（< {tau:.6f}），跳过该层，继续尝试下一层")

    for m in bn_modules:
        m.train()

    final_acc = compute_accuracy(model, val_loader, device)
    print(f"\n最终验证精度: {final_acc:.2f}%")
    print(f"激活层数: {len(activated)}/{len(queue)}")
    return activated


def client_adaptive_activation(model, train_loader, val_loader, device,
                               lora_rank=8, tau_percentile=20,
                               pilot_epochs=8, epochs_per_layer=10,
                               pilot_lr=5e-4, activation_lr=5e-4,
                               min_layers=6, max_layers=16,
                               warmup_epochs=30, warmup_lr=5e-3):
    queue = build_conv_lora_queue(model, rank=lora_rank)
    freeze_backbone(model)
    model = model.to(device)

    warmup_classifier(model, train_loader, val_loader, device,
                      max_epochs=warmup_epochs, lr=warmup_lr)
    freeze_backbone(model)
    for param in model.fc.parameters():
        param.requires_grad = False

    clean_state = {k: v.clone() for k, v in model.state_dict().items()}

    tau, contributions = auto_calibrate_threshold(
        model, queue, train_loader, val_loader,
        p=tau_percentile, pilot_epochs=pilot_epochs, lr=pilot_lr, device=device
    )
    print(f"  校准阈值 τ = {tau:.6f}")

    model.load_state_dict(clean_state)
    for lora in queue:
        lora.deactivate()

    activated = early_stop_activation(
        model, queue, train_loader, val_loader,
        tau=tau, epochs_per_layer=epochs_per_layer, lr=activation_lr, device=device
    )

    if len(activated) < min_layers and len(contributions) > 0:
        sorted_idx = sorted(range(len(contributions)),
                           key=lambda i: contributions[i], reverse=True)
        for idx in sorted_idx:
            if idx not in activated:
                queue[idx].activate()
                activated.append(idx)
                if len(activated) >= min_layers:
                    break
        activated = sorted(activated)
        print(f"  [保底] 激活层数不足，强制补齐至 {len(activated)} 层: {activated}")

    if len(activated) > max_layers:
        for i in range(max_layers, len(activated)):
            queue[activated[i]].deactivate()
        activated = activated[:max_layers]

    return model, queue, activated
