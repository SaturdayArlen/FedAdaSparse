import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def compute_model_size_bytes(model):
    total_bytes = 0
    for param in model.parameters():
        if param.requires_grad:
            total_bytes += param.numel() * param.element_size()
    return total_bytes


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_one_epoch_with_proximal(model, train_loader, optimizer, criterion, device,
                                 initial_params, mu=0.01):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        ce_loss = criterion(outputs, labels)
        proximal_term = 0.0
        for w, w_init in zip(model.parameters(), initial_params):
            proximal_term += (w - w_init).norm(2)
        loss = ce_loss + (mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def warmup_classifier(model, train_loader, val_loader, device,
                      max_epochs=30, lr=5e-3, patience=5):
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.fc.state_dict())
    no_improve = 0

    print("预热分类头...")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = compute_loss(model, val_loader, criterion, device)
        print(f"  预热 Epoch {epoch}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.fc.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停：连续 {patience} 轮无改善。")
                break

    model.fc.load_state_dict(best_state)
    print(f"预热完成，最佳验证损失: {best_val_loss:.4f}")

    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False
    return best_val_loss
