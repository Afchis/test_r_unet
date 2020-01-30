import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, depth, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    #print(pred.shape, target.shape, depth.shape)
    intersection = (pred * target * depth).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / ((pred*depth).sum(dim=2).sum(dim=2) + (target*depth).sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(prediction, target, depth, bce_weight=0.3):
    try:
        prediction = prediction.reshape(timesteps*batch_size, 1, input_size, input_size)
        target = target.reshape(timesteps*batch_size, 1, input_size, input_size)
        depth = depth.reshape(timesteps*batch_size, 1, input_size, input_size)
    except RuntimeError:
        prediction = prediction.reshape(timesteps*1, 1, input_size, input_size) # last_batch = 1
        target = target.reshape(timesteps*1, 1, input_size, input_size)
        depth = depth.reshape(timesteps*1, 1, input_size, input_size)
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target, depth)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss