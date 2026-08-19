from typing import Tuple

import torch
from torch import nn as nn
from torch.nn.functional import l1_loss

from projects.mmdet3d_plugin.models.utils.map_utils import normalize_2d_pts

from mmdet.models.builder import LOSSES


@torch.jit.script
def _compute_intra_geometrics(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset_x = x - torch.roll(x, shifts=1, dims=1)
    rl_offset_x = torch.roll(offset_x, shifts=1, dims=1)

    length = torch.norm(offset_x, p=2, dim=-1)
    rl_length = torch.norm(rl_offset_x, p=2, dim=-1)
    denom = length * rl_length + 1e-6

    dots = torch.sum(offset_x * rl_offset_x, dim=-1) / denom
    crosses = (
        offset_x[..., 0] * rl_offset_x[..., 1]
        - offset_x[..., 1] * rl_offset_x[..., 0]
    ) / denom

    return length.flatten(), dots.flatten(), crosses.flatten()


@torch.jit.script
def _compute_inter_geometrics(
    x: torch.Tensor,
    offset_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = x.shape[0]
    if N == 0:
        empty = x.new_zeros((0,))
        return empty, empty, empty

    pair_mask = torch.triu(
        torch.ones((N, N), dtype=torch.bool, device=x.device), diagonal=1
    )
    if not torch.any(pair_mask):
        empty = x.new_zeros((0,))
        return empty, empty, empty

    x_src = x.unsqueeze(1).expand(-1, N, -1, -1)
    x_tgt = x.unsqueeze(0).expand(N, -1, -1, -1)
    diff = x_src.unsqueeze(3) - x_tgt.unsqueeze(2)
    length = torch.norm(diff, p=2, dim=-1)

    offset_src = offset_x.unsqueeze(1).expand(-1, N, -1, -1)
    offset_tgt = offset_x.unsqueeze(0).expand(N, -1, -1, -1)
    offset_norm_src = torch.norm(offset_src, p=2, dim=-1)
    offset_norm_tgt = torch.norm(offset_tgt, p=2, dim=-1)
    denom = offset_norm_src.unsqueeze(-1) * offset_norm_tgt.unsqueeze(-2) + 1e-6

    dots = torch.einsum('abpd,abqd->abpq', offset_src, offset_tgt) / denom
    cross = (
        offset_src[..., 0].unsqueeze(-1) * offset_tgt[..., 1].unsqueeze(-2)
        - offset_src[..., 1].unsqueeze(-1) * offset_tgt[..., 0].unsqueeze(-2)
    ) / denom

    mask = pair_mask.unsqueeze(-1).unsqueeze(-1).expand_as(length)
    mask_f = mask.to(length.dtype)
    length = length * mask_f
    dots = dots * mask_f
    cross = cross * mask_f
    # length = length.masked_select(mask)
    # dots = dots.masked_select(mask)
    # cross = cross.masked_select(mask)

    return length.reshape(-1), dots.reshape(-1), cross.reshape(-1)


def _finite_l1_loss(input, target, size_average=None, reduction='mean'):
    """L1 reduction over finite targets without materializing index tensors."""
    if size_average is not None or reduction not in ('mean', 'sum'):
        finite = torch.isfinite(target)
        return l1_loss(input[finite], target[finite], size_average,
                       reduction=reduction)

    finite = torch.isfinite(target)
    difference = torch.where(
        finite, torch.abs(input - target), torch.zeros_like(input))
    total = difference.sum()
    if reduction == 'sum':
        return total

    count = finite.sum()
    finite_mean = total / count.clamp_min(1)
    # Match l1_loss(empty, empty, reduction='mean'): NaN value with a
    # zero gradient. The branch stays on-device and does not synchronize.
    empty_mean = total * 0.0 + input.new_tensor(float('nan'))
    return torch.where(count > 0, finite_mean, empty_mean)

@LOSSES.register_module()
class GeometricLoss(nn.Module):
    """
        Implementation of Geometric Loss
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        intra_loss_weight (float, optional): The weight of Euclidean shape loss.
        inter_loss_weight (float, optional): The weight of Euclidean relation loss.
        num_ins (int, optional): The number of instances.
        num_pts (int, optional): The number of fixed points of each instance.
            --------------------------------------------------
            The indices of predictions is organized as follows:

            Instance 0: {0, 1, ..., num_pts-1}
            Instance 1: {num_pts, num_pts+1, ..., 2*num_pts-1}
            ...
            Instance num_ins-1: {(num_ins-1)*num_pts, (num_ins-1)*num_pts+1, ..., num_ins*num_pts-1}
            ---------------------------------------------------
        num_classes (int, optional): The number of instance categories
            "num_classes + 1" is adopted to mark prediction which is matched to no gt.
        pc_range (list[float], optional): The range of lidar point clouds, formated as follows:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        loss_type (str, optional): The type of loss to measure dicrepancies between preds and gt.
            Options are "l1".   
    """

    def __init__(
            self, reduction='mean', loss_weight=1.0, 
            intra_loss_weight=1.0, inter_loss_weight=1.0,
            num_ins=50, num_pts=20, 
            num_classes=3,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            loss_type='l1',
        ):
        super(GeometricLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.inter_loss_weight = inter_loss_weight
        self.num_ins = num_ins
        self.num_pts = num_pts
        self.num_classes = num_classes
        self.pc_range = pc_range
        if loss_type == 'l1':
            self.loss_geo = l1_loss
        else:
            raise NotImplementedError('Only "l1" is supported as "loss_type".')
    
    @staticmethod
    def batch_cross_product(a, b):
        ax, ay = a[:, :, 0], a[:, :, 1]
        bx, by = b[:, :, 0], b[:, :, 1]
        return torch.mul(ax, by) - torch.mul(bx, ay)
    
    @staticmethod
    def batch_dot_product(a, b):
        return torch.sum(torch.mul(a, b), dim=-1)
    
    def compute_intra_geometrics(self, x):
        return _compute_intra_geometrics(x)
    
    def compute_inter_geometrics(self, x, offset_x):
        return _compute_inter_geometrics(x, offset_x)

    def compute_offset_geometrics(self, x):
        offset_x = torch.roll(x, shifts=-1, dims=1) - x
        inv_offset_x = torch.roll(x, shifts=1, dims=1) - x
        offset_x[:, -1] = torch.tensor([0, 0], dtype=x.dtype, device=x.device)
        inv_offset_x[:, 0] = torch.tensor([0, 0], dtype=x.dtype, device=x.device)
        offset = torch.cat([offset_x, inv_offset_x], dim=-1)
        return offset

    def forward(self,
                pred,
                target,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                valid_mask=None):
        """
            Forward function.
        Args:
            pred (torch.Tensor): The prediction of shape (B * N, Nv, 2)
                ----------------------------------------------
                B: batch size
                N: the number of instances of each sample
                Nv: the number points on each instance 
                ----------------------------------------------
            target (torch.Tensor): The learning target of the prediction of shape (B * N, Nv, 2)
            labels (torch.Tensor): The predicted labels of each instance
                ----------------------------------------------
                "self.num_classes + 1" is adopted to mark the prediction which is matched to no gt.
                The geometric loss requires this information to avoid contribution from unmatched 
                instances, especially in Euclidean relation loss.
                ----------------------------------------------
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        normalized_target = normalize_2d_pts(target, self.pc_range)
        # normalize target points to [-1, 1]
        # intra, shape
        intra_loss = pred.new_tensor(0.0)
        if torch.any(valid_mask):
            ft_preds = pred[valid_mask][..., :2]
            # ft_preds_offset = pred[valid_mask][..., 2:]
            ft_targets = normalized_target[valid_mask]
            # ft_targets_denormalized = target[..., :2][valid_mask]
            length_preds, dot_preds, cross_preds = self.compute_intra_geometrics(ft_preds)
            length_targets, dot_targets, cross_targets = self.compute_intra_geometrics(ft_targets)
            # offset_targets = self.compute_offset_geometrics(ft_targets_denormalized)
            intra_loss += _finite_l1_loss(
                length_preds, length_targets, weight, reduction)
            intra_loss += _finite_l1_loss(
                dot_preds, dot_targets, weight, reduction)
            intra_loss += _finite_l1_loss(
                cross_preds, cross_targets, weight, reduction)
        # isnotnan = torch.isfinite(offset_targets)
        # intra_loss += self.loss_geo(ft_preds_offset[isnotnan], offset_targets[isnotnan], weight, reduction=reduction)

        # inter, relation
        num_pts = pred.size(1)
        inter_loss = pred.new_tensor(0.0)
        re_preds = pred[..., :2].view(-1, self.num_ins, num_pts, 2)
        re_targets = normalized_target.view(-1, self.num_ins, num_pts, 2)
        re_labels = labels.view(-1, self.num_ins)
        valid_mask = valid_mask.view(-1, self.num_ins)

        offset_re_preds = re_preds - torch.roll(re_preds, shifts=1, dims=2)
        offset_re_targets = re_targets - torch.roll(re_targets, shifts=1, dims=2)
        for idx in range(re_preds.shape[0]):
            if torch.any(valid_mask[idx]):
                ft_preds = re_preds[idx][valid_mask[idx]]
                ft_targets = re_targets[idx][valid_mask[idx]]
                ft_offset_preds = offset_re_preds[idx][valid_mask[idx]]
                ft_offset_targets = offset_re_targets[idx][valid_mask[idx]]
                length_preds, dot_preds, cross_preds = self.compute_inter_geometrics(ft_preds, ft_offset_preds)
                length_targets, dot_targets, cross_targets = self.compute_inter_geometrics(ft_targets, ft_offset_targets)
                inter_loss += _finite_l1_loss(
                    length_preds, length_targets, weight, reduction)
                inter_loss += _finite_l1_loss(
                    dot_preds, dot_targets, weight, reduction)
                inter_loss += _finite_l1_loss(
                    cross_preds, cross_targets, weight, reduction)
        if self.inter_loss_weight is not None or self.intra_loss_weight is not None:
            loss = pred.new_tensor(0.0)
            if self.inter_loss_weight is not None:
                loss += self.inter_loss_weight * inter_loss
            else:
                loss += inter_loss
            if self.intra_loss_weight is not None:
                loss += self.intra_loss_weight * intra_loss
            else:
                loss += intra_loss
        else:
            loss = intra_loss + inter_loss
        # import pdb; pdb.set_trace()
        return loss * self.loss_weight

@LOSSES.register_module()
class PivotsGeometricLoss(GeometricLoss):
    """
        Implementation of Geometric Loss
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        intra_loss_weight (float, optional): The weight of Euclidean shape loss.
        inter_loss_weight (float, optional): The weight of Euclidean relation loss.
        num_ins (int, optional): The number of instances.
        num_pts (int, optional): The number of fixed points of each instance.
            --------------------------------------------------
            The indices of predictions is organized as follows:

            Instance 0: {0, 1, ..., num_pts-1}
            Instance 1: {num_pts, num_pts+1, ..., 2*num_pts-1}
            ...
            Instance num_ins-1: {(num_ins-1)*num_pts, (num_ins-1)*num_pts+1, ..., num_ins*num_pts-1}
            ---------------------------------------------------
        num_classes (int, optional): The number of instance categories
            "num_classes + 1" is adopted to mark prediction which is matched to no gt.
        pc_range (list[float], optional): The range of lidar point clouds, formated as follows:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        loss_type (str, optional): The type of loss to measure dicrepancies between preds and gt.
            Options are "l1".   
    """

    def __init__(self, **kwargs):
        super(PivotsGeometricLoss, self).__init__(**kwargs)

    def forward(self,
                pred,
                target,
                labels,
                pts_cls_labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """
            Forward function.
        Args:
            pred (torch.Tensor): The prediction of shape (B * N, Nv, 2)
                ----------------------------------------------
                B: batch size
                N: the number of instances of each sample
                Nv: the number points on each instance 
                ----------------------------------------------
            target (torch.Tensor): The learning target of the prediction of shape (B * N, Nv, 2)
            labels (torch.Tensor): The predicted labels of each instance
                ----------------------------------------------
                "self.num_classes + 1" is adopted to mark the prediction which is matched to no gt.
                The geometric loss requires this information to avoid contribution from unmatched 
                instances, especially in Euclidean relation loss.
                ----------------------------------------------
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        normalized_target = normalize_2d_pts(target, self.pc_range)
        # normalize target points to [-1, 1]
        # intra, shape
        intra_loss = 0
        ft_preds = pred[labels < self.num_classes][..., :2]
        ft_targets = normalized_target[labels < self.num_classes]
        ft_pivots_targets = pts_cls_labels[labels < self.num_classes]

        for i, (ft_preds_i, ft_targets_i) in enumerate(zip(ft_preds, ft_targets)):
            # 1. pick collinear pt idx
            matched_pt_mask = ft_pivots_targets[i] == 0
            matched_pt_idx = torch.nonzero(matched_pt_mask).view(-1)
            # collinear_preds_pts = ft_preds_i[~matched_pt_mask]
            ft_matched_targets_i = ft_targets_i[matched_pt_mask]
            # 2. interpolate tgt_pts
            cnt = 0
            for i in range(len(matched_pt_idx)-1):
                start_pt, end_pt = ft_matched_targets_i[i], ft_matched_targets_i[i+1]
                inter_num = matched_pt_idx[i+1] - matched_pt_idx[i] - 1
                ft_targets_i[i+1+cnt:i+1+cnt+inter_num] = self.interpolate(start_pt, end_pt, inter_num)
                cnt += inter_num

        length_preds, dot_preds, cross_preds = self.compute_intra_geometrics(ft_preds)
        length_targets, dot_targets, cross_targets = self.compute_intra_geometrics(ft_targets)
        isnotnan = torch.isfinite(length_targets)
        intra_loss += self.loss_geo(length_preds[isnotnan], length_targets[isnotnan], weight, reduction=reduction)
        isnotnan = torch.isfinite(dot_targets)
        intra_loss += self.loss_geo(dot_preds[isnotnan], dot_targets[isnotnan], weight, reduction=reduction)
        isnotnan = torch.isfinite(cross_targets)
        intra_loss += self.loss_geo(cross_preds[isnotnan], cross_targets[isnotnan], weight, reduction=reduction)

        # inter, relation
        inter_loss = 0
        re_preds = pred[..., :2].view(-1, self.num_ins, self.num_pts, 2)
        re_labels = labels.view(-1, self.num_ins)

        instance_num_list = [v[re_labels[i] < self.num_classes].shape[0] for i, v in enumerate(re_preds)]
        ft_preds = torch.split(ft_preds, instance_num_list, dim=0)
        ft_targets = torch.split(ft_targets, instance_num_list, dim=0)

        # import pdb; pdb.set_trace()
        for idx in range(len(instance_num_list)):
            ft_preds_i = ft_preds[idx]
            ft_targets_i = ft_targets[idx]
            ft_offset_preds_i = ft_preds_i - torch.roll(ft_preds_i, shifts=1, dims=1)
            ft_offset_targets_i = ft_targets_i - torch.roll(ft_targets_i, shifts=1, dims=1)

            length_preds, dot_preds, cross_preds = self.compute_inter_geometrics(ft_preds_i, ft_offset_preds_i)
            length_targets, dot_targets, cross_targets = self.compute_inter_geometrics(ft_targets_i, ft_offset_targets_i)
            isnotnan = torch.isfinite(length_targets)
            inter_loss += self.loss_geo(length_preds[isnotnan], length_targets[isnotnan], weight, reduction=reduction)
            isnotnan = torch.isfinite(dot_targets)
            inter_loss += self.loss_geo(dot_preds[isnotnan], dot_targets[isnotnan], weight, reduction=reduction)
            isnotnan = torch.isfinite(cross_targets)
            inter_loss += self.loss_geo(cross_preds[isnotnan], cross_targets[isnotnan], weight, reduction=reduction)
        if self.inter_loss_weight is not None or self.intra_loss_weight is not None:
            loss = 0
            if self.inter_loss_weight is not None:
                loss += self.inter_loss_weight * inter_loss
            else:
                loss += inter_loss
            if self.intra_loss_weight is not None:
                loss += self.intra_loss_weight * intra_loss
            else:
                loss += intra_loss
        else:
            loss = intra_loss + inter_loss
        # import pdb; pdb.set_trace()
        return loss * self.loss_weight

    def interpolate(self, start_pt, end_pt, inter_num):
        res = torch.zeros((inter_num, 2), dtype=start_pt.dtype, device=start_pt.device)
        num_len = inter_num + 1  # segment num.
        for i in range(1, num_len):
            ratio = i / num_len
            res[i-1] = (1 - ratio) * start_pt + ratio * end_pt
        return res