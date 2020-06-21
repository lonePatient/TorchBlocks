import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric=TripletDistanceMetric.EUCLIDEAN, average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.average = average
        self.distance_metric = distance_metric

    def forward(self, anchor, positive, negative):
        distance_positive = self.distance_metric(anchor, positive)
        distance_negative = self.distance_metric(anchor, negative)
        losses = F.relu(
            (distance_positive - distance_negative) + self.margin
        )
        return losses.mean() if self.average else losses.sum()
