import numpy as np
from torch.utils.data import DataLoader, Subset


def split_non_iid(dataset, num_clients, alpha=0.1, num_classes=None, val_ratio=0.2):
    if num_classes is None:
        labels = [dataset[i][1] for i in range(len(dataset))]
        num_classes = len(set(labels))

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    client_train_indices = [[] for _ in range(num_clients)]
    client_val_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx_k)).astype(int)
        proportions[-1] = len(idx_k) - np.sum(proportions[:-1])
        start = 0
        for i in range(num_clients):
            client_idx = idx_k[start:start + proportions[i]]
            split = int(len(client_idx) * (1 - val_ratio))
            train_idx = client_idx[:split]
            val_idx = client_idx[split:]
            client_train_indices[i].extend(train_idx.tolist())
            client_val_indices[i].extend(val_idx.tolist())
            start += proportions[i]

    train_loaders = []
    val_loaders = []
    for i in range(num_clients):
        train_subset = Subset(dataset, client_train_indices[i])
        val_subset = Subset(dataset, client_val_indices[i])
        train_loaders.append(DataLoader(train_subset, batch_size=32, shuffle=True))
        val_loaders.append(DataLoader(val_subset, batch_size=32, shuffle=False))
    return train_loaders, val_loaders
