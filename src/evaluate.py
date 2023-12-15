""" Model evalutation script """

import torch
from tqdm import tqdm

from src.utils import AverageMeter, accuracy

def eval_fn(model, criterion, loader, device):
    """
    Evaluation method
    :param model: model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :return: accuracy on the data
    """
    score = AverageMeter()
    losses = AverageMeter()
    model.eval()

    t = tqdm(loader)
    with torch.no_grad():
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            n = images.size(0)
            losses.update(loss.item(), n)
            score.update(acc.item(), n)

            t.set_description('(=> Eval) Accuracy: {:.4f}, Loss: {:.4f}'.format(score.avg, losses.avg))

    return score.avg, losses.avg