import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.7):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        APDistance = (anchor - positive).pow(2).sum(1)
        ANDistance = (anchor - negative).pow(2).sum(1)
        loss = F.relu(APDistance - ANDistance + self.margin)
        return loss.mean()
