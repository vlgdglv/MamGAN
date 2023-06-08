import torch.nn as nn
import torch.nn.functional as F


class Hinge(nn.Module):
    def forward(self, score_fake, score_real=None):
        if score_real is not None:
            loss_real = F.relu(1 - score_real).mean()
            loss_fake = F.relu(1 + score_fake).mean()
            return loss_fake + loss_real
        else:
            loss = - score_fake.mean()
            return loss
