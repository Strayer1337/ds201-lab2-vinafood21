import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("DataLoader không có batch nào.")
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, labels=None):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == targets).sum().item()
        total += batch_size
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    if total == 0:
        raise ValueError("DataLoader không có batch nào.")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    precision = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    return (
        running_loss / total,
        correct / total,
        precision,
        recall,
        macro_f1,
        y_true,
        y_pred,
    )

