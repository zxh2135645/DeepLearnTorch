import torch.nn as nn
import torch.nn.functional as F
import torch


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets, mask):
        # logits = logits * mask
        targets = targets * mask
        probs = F.sigmoid(logits)
        probs = probs * mask
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits) # No thresholding probs from 0-1
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        smooth = 1e-8
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, logits, targets, mask):
        probs = F.sigmoid(logits)
        probs = probs * mask
        # logit = logit.clamp(self.eps, 1. - self.eps)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # I'm assuming targets_flat is binary
        loss = -1 * self.alpha * (targets_flat * torch.log(probs_flat+self.eps) + (1-targets_flat) * torch.log(1-probs_flat+self.eps)) # cross entropy
        loss = loss * (1 - probs_flat) ** self.gamma  # focal loss

        return loss.sum()