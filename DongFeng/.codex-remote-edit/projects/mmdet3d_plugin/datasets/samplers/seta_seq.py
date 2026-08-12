import math
import copy
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
import torch.distributed as dist
from sklearn.cluster import KMeans
from typing import List, Any, Optional
import warnings
import random
from mmcv.runner import get_dist_info

from torch.utils.data import Sampler
from .sampler import SAMPLER
from .group_sampler import sync_random_seed

@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

class FlagBasedSeTa(Dataset):
    """
    FlagBasedSeTa 实现了基于flag分组的SeTa采样策略, 对每个clip的所有样本loss均值进行减枝
    """
    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        prune_ratio: float = 0.5,
        num_group: int = 10,
        window_scale: float = 0.5,
        delta: float = 0.875,
        flag_key: str = 'flag'  # 用于分组的flag属性名
    ):
        super(FlagBasedSeTa, self).__init__()
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        self.num_group = num_group
        self.window_scale = window_scale
        self.flag_key = flag_key
        
        # 确保数据集有flag属性
        assert hasattr(self.dataset, flag_key), f"Dataset must have a '{flag_key}' attribute for grouping"
        self.flag = getattr(self.dataset, flag_key)
        self.group_indices = self._get_group_indices()
        
        # 存储每个clip(group)的loss均值
        self.group_losses = torch.ones(len(self.group_indices)) * 5
        self.weights = torch.ones(len(self.dataset)) * 5
        self.num_pruned_samples = 0
        self.iterations = 0
        self.cur_epoch = 1
        self.cur_batch_indices = None
        self.unique_flags = None
        self.sample_indices = self.group_indices
        
    def _get_group_indices(self) -> List[List[int]]:
        """根据flag将数据集索引分组"""
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.select_group_id = [i for i in range(self.groups_num)]
        group_idx_to_sample_idxs = [[] for _ in range(self.groups_num)]
        for idx, group_idx in enumerate(self.flag):
            group_idx_to_sample_idxs[group_idx].append(idx)
        return group_idx_to_sample_idxs
    
    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        """设置当前批次的索引"""
        self.cur_batch_indices = cur_batch_indices
    

    def update(self, values: torch.Tensor) -> None:
        """
        更新每个clip的平均损失值
        
        参数:
            values: 批次中每个样本的损失值，形状应为 [batch_size]
        """
        # 验证输入
        assert isinstance(values, torch.Tensor), "输入必须是PyTorch张量"
        values = torch.nan_to_num(values)
        batch_size = len(self.cur_batch_indices)
        
        values = torch.stack([values.detach()] * batch_size, dim=0)
        device = values.device
        indices = self.cur_batch_indices.to(device)
        loss_val = values.detach().clone()

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1), loss_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            loss_val = iv_whole_group[1]
        
        # 保存当前批次样本的损失值
        self.weights[indices.cpu().long()] = loss_val.cpu()
        indices = indices.cpu().long().tolist()
        # 获取当前批次中每个样本所属的clip标识
        unique_flags = [self.flag[idx] for idx in indices]
        # print("flag", unique_flags)
        # print("cur_batch_indices", indices)
        
        # 重置当前批次索引
        self.cur_batch_indices = []

    def _update_group_losses(self) -> None:
        """更新每个clip的平均损失值（辅助方法）"""
        # 对每个唯一的clip标识，计算其平均损失
        for flag,flag_indices in zip(self.select_group_id, self.sample_indices):
            # 计算这些样本的平均损失
            avg_loss = self.weights[flag_indices].mean()
            # 更新该clip的平均损失
            self.group_losses[flag] = avg_loss

    # def update(self, values):
    #     """
    #     更新clip的loss均值
    #     values: 批次样本的loss值
    #     """
    #     assert isinstance(values, torch.Tensor)
    #     device = values.device
    #     # import ipdb 
    #     # ipdb.set_trace()
    #     # 获取当前批次中每个样本所属的clip
    #     unique_flags = [self.flag[idx] for idx in self.cur_batch_indices]
        
    #     batch_size = len(self.cur_batch_indices)
    #     values = torch.stack([values.detach()] * batch_size, dim=0)

    #     self.weights[self.cur_batch_indices] = values.detach().cpu()
    #     # 计算每个clip的loss均值
    #     if self.unique_flags is not None:
    #         self.unique_flags = unique_flags
    #     else:
    #         for idx, flag in enumerate(unique_flags):
    #             if flag != self.unique_flags:
    #                 same_flag_index = np.where(self.flag == 0)[0]
    #                 self.group_losses[flag] = self.weights[same_flag_index].mean()

    #     self.unique_flags = unique_flags

    #     self.cur_batch_indices = []
        
    #     # 分布式环境下同步group_losses
    #     if dist.is_available() and dist.is_initialized():
    #         world_size = dist.get_world_size()
    #         group_losses_list = [torch.ones_like(self.group_losses) for _ in range(world_size)]
    #         dist.all_gather(group_losses_list, self.flag)
            
    #         # 合并所有进程的group_losses
    #         self.group_losses = torch.stack(group_losses_list).mean(dim=0)

    def _apply_random_drop(self, indices):
        """应用随机丢弃样本策略"""
        drop_list = getattr(self, 'drop_list', [0.05, 0.35, 0.65, 0.9, 1])
        if len(drop_list) != 5:
            raise ValueError('drop_list should be of length 5')
            
        new_indices = []
        for idx in indices:
            random_prob = random.uniform(0, 1)
            drop_level = 0
            
            # 确定丢弃级别
            for i, threshold in enumerate(drop_list):
                if random_prob <= threshold:
                    drop_level = i
                    break
                    
            # 根据丢弃级别决定是否保留样本
            if drop_level == 0:
                new_indices.append(idx)
        
        # 如果全部被丢弃，至少保留一个样本
        if not new_indices and indices:
            new_indices = [indices[0]]
        
        yield new_indices


    def prune(self):
        """基于clip的loss均值进行SeTa剪枝"""
        seed = sync_random_seed()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 按保留比例随机选择clip
        group_scores = self.group_losses.clone()
        group_indices = torch.arange(len(self.group_indices))
        num_kept_groups = int(len(self.group_indices) * self.keep_ratio)
        
        if num_kept_groups <= 0:
            num_kept_groups = 1  # 至少保留一个组
        
        # 随机选择保留的组
        perm = torch.randperm(len(self.group_indices))
        selected_group_indices = perm[:num_kept_groups]
        selected_group_scores = group_scores[selected_group_indices]
        selected_groups = [self.group_indices[i] for i in selected_group_indices]
        
        # 2. 对保留的组进行KMeans聚类
        if len(selected_groups) > self.num_group:
            grouped_scores, grouped_indices = self._kmeans_group(selected_group_scores, selected_group_indices)
        else:
            # 组数不足时直接分组
            grouped_scores = [selected_group_scores[i:i+1] for i in range(len(selected_group_scores))]
            grouped_indices = [selected_group_indices[i:i+1] for i in range(len(selected_group_indices))]
        
        # 3. 滑动窗口从简单到困难选择组
        selected_grouped_indices = self._slide_easy2hard(grouped_indices, self.iterations)
        selected_groups = [self.group_indices[i] for group in selected_grouped_indices for i in group]
        self.select_group_id = [i for group in selected_grouped_indices for i in group]
           
        # 打印剪枝信息
        # self._print_prune_info(selected_group_scores, grouped_scores, selected_indices)
        
        return selected_groups
    
    def _kmeans_group(self, scores: torch.Tensor, indices: torch.Tensor) -> (List[torch.Tensor], List[torch.Tensor]):
        """对组进行KMeans聚类"""
        if len(scores) <= self.num_group:
            return [scores], [indices]
        
        kmeans = KMeans(n_clusters=self.num_group, random_state=0).fit(scores.numpy().reshape(-1, 1))
        labels = kmeans.labels_
        
        grouped_scores = [[] for _ in range(self.num_group)]
        grouped_indices = [[] for _ in range(self.num_group)]
        
        for score, index, label in zip(scores, indices, labels):
            grouped_scores[label].append(score.item())
            grouped_indices[label].append(index.item())
        
        # 按聚类中心排序
        group_centers = [np.mean(group) for group in grouped_scores]
        sorted_groups = sorted(zip(group_centers, grouped_scores, grouped_indices), key=lambda x: x[0])
        
        sorted_grouped_scores = [torch.tensor(group[1]) for group in sorted_groups]
        sorted_grouped_indices = [torch.tensor(group[2]) for group in sorted_groups]
        
        return sorted_grouped_scores, sorted_grouped_indices
    
    def _slide_easy2hard(self, grouped_indices: List[torch.Tensor], cur_iterations: int) -> List[torch.Tensor]:
        """从简单到困难滑动选择组"""
        if cur_iterations == 0:
            return grouped_indices
        
        num_group = len(grouped_indices)
        window_size = max(1, round(num_group * self.window_scale))
        slide_size = num_group - window_size
        
        if slide_size <= 0:
            return grouped_indices
        
        start = cur_iterations % (slide_size + 1)
        end = start + window_size
        return grouped_indices[start:end]
    
    # def _print_prune_info(self, selected_group_scores: torch.Tensor, grouped_scores: List[torch.Tensor], selected_indices: List[int]):
    #     """打印剪枝信息"""
    #     print('\n|--| SeTa Pruning Statistics')
    #     print(f'|--| Total groups: {len(self.group_indices)}, kept groups: {len(selected_group_scores)}')
        
    #     if len(grouped_scores) > 0:
    #         group_sizes = [len(group) for group in grouped_scores]
    #         print(f'|--| Each group size: {group_sizes}')
        
    #     saved = 1 - len(selected_indices) / len(self.dataset)
    #     print(f'|--| #sampled: {len(selected_indices)} #saved: {saved * 100:.2f}%')
    
    def no_prune(self):
        """不进行剪枝，返回所有样本"""
        seed = sync_random_seed()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 按保留比例随机选择clip
        group_scores = self.group_losses.clone()
        group_indices = torch.arange(len(self.group_indices))
        num_kept_groups = int(len(self.group_indices) * self.keep_ratio)
        
        if num_kept_groups <= 0:
            num_kept_groups = 1  # 至少保留一个组
        
        # 随机选择保留的组
        perm = torch.randperm(len(self.group_indices))
        selected_group_indices = perm[:num_kept_groups]
        selected_group_scores = group_scores[selected_group_indices]
        selected_groups = [self.group_indices[i] for i in selected_group_indices]
        self.select_group_id = selected_group_indices

        return selected_groups
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return index, self.dataset[index]
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    @property
    def stop_prune(self):
        return self.num_epochs * self.delta
    
    def get_saved_ratio(self):
        return self.num_pruned_samples / (len(self.dataset) * self.num_epochs)

    def reset_prune(self):
        """重置采样器，根据当前迭代次数决定是否进行剪枝"""
        # 更新每个clip的平均损失
        self._update_group_losses()

        if self.cur_epoch > self.stop_prune or self.cur_epoch == 1:
            # 停止剪枝，使用所有样本
            # self.sample_indices = self.no_prune()
            self.sample_indices = self.group_indices
            self.select_group_id = [i for i in range(self.groups_num)]
        else:
            # 进行剪枝
            self.sample_indices = self.prune()

        # 展平所有选中的样本索引
        selected_indices = []
        for group in self.sample_indices:
            selected_indices.extend(group)
        
        # 统计剪枝数量
        self.num_pruned_samples += len(self.dataset) - len(selected_indices)
        self.num_select_samples = len(selected_indices)
        
        if self.cur_epoch == self.num_epochs:
            print("="*20, "Final Pruning Stats", "="*20)
            print(f"===> Saved ratio: {self.get_saved_ratio()*100:.2f}%")
            print(f"===> Pruned samples: {self.num_pruned_samples}")
            print("="*50)
        
        self.cur_epoch += 1
        self.iterations += 1

class PseudoSampler(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def set_epoch(self, epoch):
        self.epoch = epoch

@SAMPLER.register_module()
class DistributedFlagBasedSeTaSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        self.sampler = PseudoSampler(dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = samples_per_gpu * num_replicas
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        # self.group_idx_to_sample_idxs = {
        #     group_idx: np.where(self.flag == group_idx)[0].tolist()
        #     for group_idx in range(self.groups_num)}
        group_idx_to_sample_idxs = [[] for _ in range(self.groups_num)]
        for idx, group_idx in enumerate(self.flag):
            group_idx_to_sample_idxs[group_idx].append(idx)

        self.group_idx_to_sample_idxs = {group_idx: idxs for group_idx, idxs in enumerate(group_idx_to_sample_idxs)}

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def update_dataset(self):
        self.groups_num=len(self.dataset.sample_indices)
        self.group_idx_to_sample_idxs = {group_idx: idxs for group_idx, idxs in enumerate(self.dataset.sample_indices)}
        self.size = self.dataset.num_select_samples
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]

    
    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.sampler.epoch)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return (self.size + self.global_batch_size - 1) // self.global_batch_size
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    
def prune(dataset, args):

    dataset = FlagBasedSeTa(dataset, args.epochs, args.prune_ratio,
                            args.num_group, args.window_scale,
                            args.delta, args.flag_key)
    print(f'==> FlagBasedSeTa pruning: ratio={args.prune_ratio}, group={args.num_group}, window_scale={args.window_scale}')

    return dataset
