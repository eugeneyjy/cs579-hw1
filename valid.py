import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation_metrics(model, loader, criterion):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        _, pred_label = torch.max(pred, 1)

        loss = criterion(pred, target)

        sum_loss += loss.item()
        correct += (pred_label == target).sum()
        total += len(data)

    return sum_loss/total, correct/total