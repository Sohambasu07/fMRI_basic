""" Main training loop """

from tqdm import tqdm
import time

from src.utils import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    t = tqdm(loader)
    for images, labels in t:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
