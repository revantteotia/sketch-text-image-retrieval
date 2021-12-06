"""
Using image-text composition models as described in 
Composing Text and Image for Image Retrieval - An Empirical Odyssey
Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
CVPR 2019.

Code mofified from https://github.com/google/tirg for newer pytorch version.
"""

"""
Metric learning functions.
Codes are modified from:
https://github.com/google/tirg/blob/master/torch_functions.py
"""

import torch

def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    
    # print("dist shape in pairwise_distances= ", dist.shape)
    return torch.clamp(dist, 0.0, torch.inf)


class MyTripletLossFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, triplets):
        
        ctx.save_for_backward(features)
        ctx.triplets = triplets

        distances = pairwise_distances(features)
        ctx.distances = distances

        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in triplets:
            w = 1.0
            triplet_count += w
            curr_loss = w * torch.log(1 +
                                torch.exp(distances[i, j] - distances[i, k]))
              
            loss += curr_loss

            if distances[i, j] < distances[i, k]:
                correct_count += 1

        loss /= triplet_count

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        features, = ctx.saved_tensors
        features_np = features
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features

        triplets = ctx.triplets
        distances = ctx.distances
        triplet_count = len(triplets)

        for i, j, k in triplets:
            w = 1.0
            f = 1.0 - 1.0 / (
                1.0 + torch.exp(distances[i, j] - distances[i, k]))
            grad_features_np[i, :] += w * f * (
                features_np[i, :] - features_np[j, :]) / triplet_count
            grad_features_np[j, :] += w * f * (
                features_np[j, :] - features_np[i, :]) / triplet_count
            grad_features_np[i, :] += -w * f * (
                features_np[i, :] - features_np[k, :]) / triplet_count
            grad_features_np[k, :] += -w * f * (
                features_np[k, :] - features_np[i, :]) / triplet_count

        # for i in range(features_np.shape[0]):
        #     grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= grad_output

        return grad_features, None


class TripletLoss(torch.nn.Module):
    """Class for the triplet loss."""
    def __init__(self, pre_layer=None):
        super(TripletLoss, self).__init__()
        self.pre_layer = pre_layer

    def forward(self, x, triplets):
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        loss = MyTripletLossFunc.apply(x, triplets)
        return loss


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""
    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features