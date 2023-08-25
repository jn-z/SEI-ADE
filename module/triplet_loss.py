import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb

def calc_cdist(a, b, metric='cosdistance'):
    a_new = a[:, None, :] 
    b_new = b[None, :, :]
    diff = a_new - b_new
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=2)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=2)
    elif metric == 'cosdistance':
        a_new_sa = a_new.repeat(1,b_new.shape[1],1)
        b_new_sa = b_new.repeat(a_new.shape[0],1,1)
        CosineDistance = nn.CosineSimilarity(dim=2,eps=1e-6)
        cose = CosineDistance(a_new_sa, b_new_sa)
        return 1.0-cose
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


def _apply_margin(x, m):
    # pdb.set_trace()
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)


def batch_hard(cdist, pids, margin):
    """Computes the batch hard loss as in arxiv.org/abs/1703.07737.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
    """
    mask_pos = (pids[None, :] == pids[:, None]).float()

    ALMOST_INF = 9999.9
    furthest_positive = torch.max(cdist * mask_pos, dim=0)[0]
    furthest_negative = torch.min(cdist + ALMOST_INF*mask_pos, dim=0)[0]
    #furthest_negative = torch.stack([
    #    torch.min(row_d[row_m]) for row_d, row_m in zip(cdist, mask_neg)
    #]).squeeze() # stacking adds an extra dimension

    val_loss = _apply_margin(furthest_positive - furthest_negative, margin)
    return val_loss.mean()#F.relu(val_loss)

def get_valid_triplets_mask(labels,gpu):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    
    indices_equal = indices_equal.cuda(gpu) if labels.is_cuda else indices_equal
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).byte()
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k
    # pdb.set_trace()
    mask = distinct_indices & valid_labels
    return mask
def batch_all(cdist,pids, gpu, margin):
    """
    get triplet loss for all valid triplets and average over those triplets whose loss is positive.
    """
    # pdb.set_trace()
    anchor_positive_dist = cdist.unsqueeze(2)
    anchor_negative_dist = cdist.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # get a 3D mask to filter out invalid triplets
    mask = get_valid_triplets_mask(pids,gpu)

    triplet_loss = triplet_loss * mask.float()
    triplet_loss.clamp_(min=0)

    # count the number of positive triplets
    epsilon = 1e-16
    num_positive_triplets = (triplet_loss > 0).float().sum()
    num_valid_triplets = mask.float().sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)
    return triplet_loss, fraction_positive_triplets