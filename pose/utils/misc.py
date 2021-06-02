from __future__ import absolute_import

import os
import shutil
import torch
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state,  is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def lr_poly(base_lr, epoch, max_epoch, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(epoch) / max_epoch) ** power)


def adjust_learning_rate_main(optimizer, epoch, args):
    lr = lr_poly(args.lr, epoch, args.max_epoch, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
