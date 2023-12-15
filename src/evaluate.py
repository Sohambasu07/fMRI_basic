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
    model.eval()

    t = tqdm(loader)
    with torch.no_grad():
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            score.update(acc.item(), images.size(0))

            t.set_description('(=> Test) Accuracy: {:.4f}'.format(score.avg))
            t.set_description('(=> Test) Loss: {:.4f}'.format(loss))

    return score.avg, loss