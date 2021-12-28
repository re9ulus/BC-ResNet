import torchaudio
from torch import nn
import torch.nn.functional as F


def train_epoch(model, optimizer, train_loader, device, epoch, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)

        target = target.to(device)

        # TODO: Apply transform in train_loader or collate_fn
        # data = transform(data)
        output = model(data)

        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")

        losses.append(loss.item())

    return losses


def train(n_epoch, model, optimizer, train_loader, device, log_interval, scheduler=None):
    print(f"--- Start train {n_epoch} epoches")
    for epoch in range(n_epoch):
        print(f"--- Start epoch {epoch+1}")
        train_epoch(model, optimizer, train_loader, device, epoch, log_interval)
        if scheduler:
            scheduler.step()
    print("--- Done train")
