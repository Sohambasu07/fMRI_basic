""" Write additional utilities here """

import torch
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == labels) / len(labels)

def disp_image(img, label):
    plt.figure()
    plt.title(f"Label: {label}")
    plt.imshow(np.asarray(img))
    plt.show()