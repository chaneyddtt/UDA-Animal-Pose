
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, num_keypoints):
        batch_size = output.size(0)
        heatmaps_pred = output.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_keypoints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_keypoints


class CurriculumLoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(CurriculumLoss, self).__init__()
        self.criterion = nn.MSELoss(reduce=False)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, top_k):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))

        if self.use_target_weight:
            loss = 0.5 * (self.criterion(
                heatmaps_pred.mul(target_weight),
                heatmaps_gt.mul(target_weight)
            )).mean(-1)
        else:
            loss = 0.5 * (self.criterion(heatmaps_pred, heatmaps_gt)).mean(-1)
        weights_bool = (target_weight > 0)
        loss_clone = loss.clone().detach().requires_grad_(False)
        loss_inf = 1e8 * torch.ones_like(loss_clone, requires_grad=False)
        # set the loss of invalid joints (weights equal 0) to a large value such that it won't be
        # selected as reliable pseudo labels, only joints with smaller loss will be selected
        loss_clone = torch.where(weights_bool.squeeze(-1), loss_clone, loss_inf)
        _, topk_idx = torch.topk(loss_clone, k=top_k, dim=-1, largest=False)
        tmp_loss = torch.gather(loss, dim=-1, index=topk_idx)
        tmp_loss = tmp_loss.sum()/(top_k * batch_size)
        return tmp_loss


# take from the source code of the mean teacher network
def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    loss = ((input1 - input2) ** 2).mean()
    return 0.5 * loss


