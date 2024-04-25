import torch
from tqdm import tqdm

import torch.nn.functional as F


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, train_stats):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}"
        )

    train_loss = train_loss/processed
    if "loss" in train_stats:
        train_stats["loss"].append(train_loss)
    else:
        train_stats["loss"] = [train_loss]

    if "acc" in train_stats:
        train_stats["acc"].append(100 * correct / processed)
    else:
        train_stats["acc"] = [100 * correct / processed]


def test_epoch(model, device, test_loader, criterion, test_stats):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(
                dim=1, keepdim=True
            ) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if "loss" in test_stats:
        test_stats["loss"].append(test_loss)
    else:
        test_stats["loss"] = [test_loss]

    # Print test set results
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    if "acc" in test_stats:
        test_stats["acc"].append(100.0 * correct / len(test_loader.dataset))
    else:
        test_stats["acc"] = [100.0 * correct / len(test_loader.dataset)]
