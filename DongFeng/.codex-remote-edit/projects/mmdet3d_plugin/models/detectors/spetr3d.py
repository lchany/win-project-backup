# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import os
import torch
import cv2
import math
import shutil
import numpy as np
import random
import copy
import shutil
# from mmcv.runner import get_dist_info
from mmcv.runner import force_fp32, auto_fp16
from mmcv.parallel import DataContainer as DC
from mmdet.models import DETECTORS
# from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models import build_head, build_backbone, build_neck
from mmdet3d.ops import Voxelization, DynamicScatter

from torch import nn

from projects.mmdet3d_plugin.models.builder import build_query_generator
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from torch.nn.modules.batchnorm import _BatchNorm
from projects.mmdet3d_plugin.models.utils.positional_encoding import PositionEncoder
from scipy.interpolate import interp1d
import torch.nn.functional as F
import json
from mmdet.models.utils import build_transformer
from projects.mmdet3d_plugin.models.utils.builder import build_fuser
from projects.mmdet3d_plugin.core.post_processing.map_seg_post_processor import MapSegPostProcessor
from copy import deepcopy
from projects.mmdet3d_plugin.models.utils.bevformer_encoder import BEVFormerEncoderStream
from tools.model_deployments.common import save_tensor
#随机性固定
def tensor_hash(x):
    import hashlib
    import numpy as np
    import torch

    if hasattr(x, "tensor"):
        x = x.tensor
    if torch.is_tensor(x):
        x = x.detach().cpu().contiguous().numpy()
    elif isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
    else:
        return str(type(x))
    return hashlib.md5(x.tobytes()).hexdigest()[:12]
#随机性固定
def print_shape(name, var):
    if var is None:
        print(f"{name}: None")
    elif hasattr(var, "shape"):  # torch.Tensor / numpy.ndarray
        print(f"{name}: {type(var)}, shape={var.shape}, dtype={getattr(var, 'dtype', None)}")
    elif isinstance(var, (list, tuple)):
        print(f"{name}: {type(var)}, len={len(var)}")
        if len(var) > 0 and hasattr(var[0], "shape"):
            print(f"    first element shape={var[0].shape}")
    elif isinstance(var, dict):
        print(f"{name}: dict with keys={list(var.keys())}")
    else:
        print(f"{name}: {type(var)} value={var}")

MAP_KEY_MAPPING = {
    "map_all_cls_scores": "map_scores",
    "map_all_pts_preds": "map_coords",
    "map_all_pts_types": "map_types",
    "map_all_pts_colors": "map_colors",
    "map_all_pts_uncertains": "map_coords_std",
}

def save_bin_and_shape(tensor, prefix, idx):
    """
    保存 tensor 到 prefix/0.bin 和 prefix/0.shape
    """
    arr = tensor.detach().cpu().numpy().astype("float32")

    # prefix 作为目录
    os.makedirs(prefix, exist_ok=True)

    # 存 bin
    bin_path = os.path.join(prefix, f"{idx}.bin")
    arr.tofile(bin_path)

    # # 存 shape
    # shape_path = os.path.join(prefix, "0.shape")
    # with open(shape_path, "w") as f:
    #     f.write(" ".join(map(str, tensor.shape)))

    print(f"[BIN] Saved to {bin_path}")

def save_tensor_if_valid(tensor, pt_path=None, bin_prefix=None, idx=0):
    """
    保存 tensor 到 .pt 和 .bin/.shape，前提是 tensor 有效且文件不存在。
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"[SKIP] Not a tensor: {pt_path or bin_prefix}")
        return
    if tensor.numel() == 0:
        print(f"[SKIP] Empty tensor: {pt_path or bin_prefix}")
        return
    if pt_path and not os.path.exists(pt_path):
        torch.save(tensor, pt_path)
        print(f"[PT] Saved to {pt_path}")
    if bin_prefix and not os.path.exists(bin_prefix + ".bin"):
        save_bin_and_shape(tensor, bin_prefix, idx)
        print(f"[BIN] Saved to {bin_prefix}.bin / .shape")
        
def load_data(data_path, dtype, *shape):
    data = np.fromfile(data_path, dtype=dtype, sep=',')
    data = data.reshape(*shape)
    return data

def load_img(data_path):
    data = np.fromfile(data_path, dtype=np.uint8)
    data = data[::2]
    data = data.reshape(576,1024,3)
    img_data=torch.tensor(data) 

    # img_data=img_data.reshape(IMG_H,IMG_W,3)
    # BGR 2 RGB

    r=img_data[:,:,2].clone()
    img_data[:,:,2] = img_data[:,:,0]
    img_data[:,:,0] = r

    # normalize
    mean=torch.tensor([123.675, 116.28, 103.53])
    std=torch.tensor([58.395, 57.12, 57.375])
    img_data=(img_data-mean)/std

    # H W C --> B C H W
    img_data=img_data.permute(2,0,1).contiguous().reshape(1,3,576,1024)
    return img_data

def sim(t1,t2,name,ts):
    a=t1.detach().cpu().numpy()
    b=t2.detach().cpu().numpy()
    sim=np.sum(a*b)/np.sqrt(np.sum(a*a))/np.sqrt(np.sum(b*b))
    print(name,"-",ts," : ", sim)

def save_tensor(feat, save_dir, save_key, is_input=True):
    save_dir_npy = os.path.join(save_dir, "far3d_backbone_align_npy")
    os.makedirs(save_dir_npy, exist_ok=True)
    if is_input:
        save_dir_bin = os.path.join(save_dir, "far3d_backbone_align_bin_in", save_key)
    else:
        save_dir_bin = os.path.join(save_dir, "far3d_backbone_align_bin_out", save_key)
    os.makedirs(save_dir_bin, exist_ok=True)
    save_path = os.path.join(save_dir_npy, f"{save_key}.npy")
    save_ary = feat.cpu().numpy().astype(np.float32)
    np.save(save_path, save_ary)
    save_ary.tofile(os.path.join(save_dir_bin, "0.bin"))

from mmdet3d.models.builder import DETECTORS as MMDET_DETECTORS

@MMDET_DETECTORS.register_module()
class SPetr3D(MVXTwoStageDetector):
    """SPetr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_query_generator=None,
                 lane3d_head=None,
                 lane2d_head=None,
                 occ3d_head=None,
                 bev_encoder=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 pvb_pe=None,
                 lane3d_pe=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 planning_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 reset_probability=0,
                 pretrained=None,
                 fix_backbone=False,
                 fix_pvb_head=False,
                 fix_pts_backbone=False,
                 pred_frame_tf=0,
                 dynamic_collision = False,
                 bev_constraint=False,
                 seg_res=False,
                 occ_levels=[1],
                 lc_fusion=None,
                 box_pc_range=[-100.0, -50.0, -5.0, 200.0, 50.0, 3.0],
                 map_range=[-20, -24.0, -5.0, 120.0, 24.0, 3.0],
                 multimodal_gop=None,
                 pts_bbox_head_gop=None,
                 pts_query_generator_gop=None,):
        super(SPetr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.reset_probability = reset_probability
        self.fix_backbone = fix_backbone
        self.fix_pvb_head = fix_pvb_head
        self.fix_pts_backbone = fix_pts_backbone
        self.pred_frame_tf = pred_frame_tf
        self.dynamic_collision = dynamic_collision
        self.det_pred = False
        self.bev_constraint = bev_constraint
        self.occ_levels = occ_levels
        self.encoders = nn.ModuleDict()
        self.map_range=map_range
        self.box_pc_range=box_pc_range
        

#随机性固定
        self._debug_output_tensors = {}
#随机性固定
        self.frame_id=0
        if pts_bbox_head is None:
            self.pts_bbox_head = pts_bbox_head

        if bev_encoder is not None:
            self.bev_encoder = build_transformer(bev_encoder)
        else:
            self.bev_encoder = None
        self.frame_id =0 
        if pts_query_generator is not None:
            self.pts_query_generator = self.build_pts_query_generator(pts_query_generator, self.pts_bbox_head)
        self.pts_bbox_head_gop = build_head(pts_bbox_head_gop) if pts_bbox_head_gop else None
        if pts_query_generator_gop is not None:
            self.pts_query_generator_gop = self.build_pts_query_generator(pts_query_generator_gop, self.pts_bbox_head_gop)

        self.img_bev_encoder_backbone = None
        self.img_bev_encoder_neck = None
        
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = build_backbone(img_bev_encoder_backbone)
        
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)

        if planning_head is not None:
            self.planning_head = build_head(planning_head)
            self.pred_delta = planning_head.get('pred_delta', True)
        else:
            self.planning_head = None
        
        if lane2d_head is not None:
            self.lane2d_head = build_head(lane2d_head)
        else:
            self.lane2d_head = None

        if lane3d_head is not None:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            lane3d_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            lane3d_head.update(test_cfg=pts_test_cfg)
            self.lane3d_head = build_head(lane3d_head)
        else:
            self.lane3d_head = None

        if occ3d_head is not None:
            self.occ3d_head = build_head(occ3d_head)
        else:
            self.occ3d_head = None

        if pvb_pe is not None:
            self.pvb_position_encoder = PositionEncoder(**pvb_pe)
            if lane3d_pe is not None:
                self.lane3d_position_encoder = PositionEncoder(**lane3d_pe)
            else:
                self.lane3d_position_encoder = PositionEncoder(**pvb_pe)
        else:
            self.pvb_position_encoder = None
            self.lane3d_position_encoder = None
        self.seg_res = seg_res

        self.MAX_DUMP = int(os.getenv('MAX_DUMP', "1"))
        self.dump_road_count = 0
        self.dump_occ_count = 0
        #随机性固定
        self.debug_train_iter = 0
        self.debug_train_n = 10
        self.debug_print = False
        self.hit_target = False
        #随机性固定
        if multimodal_gop is not None:
            self.multimodal_gop = build_backbone(multimodal_gop)
        else:
            self.multimodal_gop = None
        
        if lc_fusion is not None:
            self.lc_fusion = build_fuser(lc_fusion)
        else:
            self.lc_fusion = None   
    
    def build_pts_query_generator(self, pts_query_generator, bbox_head):
        try:
            pts_query_generator.update(dict(
                virtual_voxel_size=self.pts_backbone.virtual_voxel_size,
                point_cloud_range=self.pts_backbone.point_cloud_range,
                head_pc_range=bbox_head.pc_range.tolist(),
            ))
        except:
            tmp_virtual_voxel_size = (
                [
                    self.pts_backbone.model_cfg.virtual_voxel_size[0] * 2,
                    self.pts_backbone.model_cfg.virtual_voxel_size[1] * 2,
                    self.pts_backbone.model_cfg.virtual_voxel_size[2],
                ]
                if self.pts_backbone.model_cfg.get("use_extra_downsample", False)
                else self.pts_backbone.model_cfg.virtual_voxel_size
            )
            pts_query_generator.update(dict(
                virtual_voxel_size=tmp_virtual_voxel_size,
                point_cloud_range=self.pts_backbone.model_cfg.point_cloud_range,
                head_pc_range=bbox_head.pc_range.tolist(),
                transformer_use_lidar_ca=bbox_head.transformer_use_lidar_ca,
            ))
        return build_query_generator(pts_query_generator)
              
    def train(self, mode: bool = True):
        r"""Sets the module in training mode, specially set norm_eval=true when debugging
        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        for module in self.children():
            module.train(mode)
            if os.getenv("DEBUG")=='True':
                for m in module.modules():
                    # trick: eval have effect on BatchNorm only
                    if isinstance(m, _BatchNorm):
                        m.eval()
        return self

    def start_det_pred(self):
        self.det_pred = True

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)     # torch.Size([2, 1, 7, 3, 544, 960])
        if img is not None:
            if img.dim() == 6:
                # enter there
                img = img.flatten(1, 2)     # torch.Size([2, 7, 3, 544, 960])
            if img.dim() == 5 and img.size(0) == 1:    # for bs=1
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)  # torch.Size([14, 3, 544, 960])
            if os.getenv("SAVE_TENSOR") == 'True' and self.frame_id == 4:
                self.save_dir = "/mnt/afs2/jinwei/codes/backbone_tensors"
                if os.path.exists(self.save_dir):
                    shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)
                all_cam=['center_camera_fov30','center_camera_fov120', 'left_front_camera', 'left_rear_camera',
                            'rear_camera', 'right_rear_camera', "right_front_camera"]
                for i in range(len(all_cam)):
                    cam=all_cam[i]
                    cur_img=img[i:i+1].clone()
                    save_tensor(cur_img, self.save_dir, cam)
            if self.use_grid_mask:  # True
                img = self.grid_mask(img)
            # np.save('align/img.npy', img.detach().cpu().numpy())
            img_feats = self.img_backbone(img)      # tuple: (torch.Size([14, 512, 17, 30]))
            if os.getenv("COMPARE") == 'True':
                ut_imgs = []
                for cam in ['center_camera_fov30','center_camera_fov120', 'left_front_camera', 'left_rear_camera',
                            'rear_camera', 'right_rear_camera', "right_front_camera"]:
                    ut_img = load_img(os.path.join("./work_dirs/nn-out/"+cam+"_"+self.timestamp)).to(img.device)
                    ut_imgs.append(ut_img)
                ut_imgs = torch.cat(ut_imgs,dim=0)
                sim(img,ut_imgs,'img',self.timestamp)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        if os.getenv("SAVE_TENSOR") == "True" and self.frame_id == 4:
            self.save_dir = "/mnt/afs2/jinwei/codes/backbone_tensors"
            os.makedirs(self.save_dir, exist_ok=True)
            all_feats=['img_feats_0', 'img_feats_1', 'img_feats_2', 'img_feats_3']
            for i in range(len(all_feats)):
                neck_feat_name = all_feats[i]
                neck_feat = img_feats[i].clone()
                save_tensor(neck_feat, self.save_dir, neck_feat_name)

        img_feats_reshapeds = []
        # B=2, len_queue=1, len(img_feats)=1
        for i in range(len(img_feats)):
            BN, C, H, W = img_feats[i].size()   # 14, 512, 17, 30
            if self.training or training_mode:
                img_feats_reshaped = img_feats[i].view(B, len_queue, int(BN/B/len_queue), C, H, W)  # torch.Size([2, 1, 7, 512, 17, 30])
            else:
                # TODO: remove hard code!!
                if self.fix_backbone and os.getenv("VIS_RATE") is None:
                    img_feats_reshaped = img_feats[i].view(B, len_queue, int(BN/B/len_queue), C, H, W)
                else:
                    img_feats_reshaped = img_feats[i].view(B, int(BN/B/len_queue), C, H, W)
            img_feats_reshapeds.append(img_feats_reshaped)
            input_keys = ['center_camera_fov30', 'center_camera_fov120', 'left_front_camera', 'left_rear_camera','rear_camera','right_rear_camera','right_front_camera']
            if os.getenv("SAVE_TENSOR") == 'True' and self.frame_id == 175:
                self.save_dir = os.getenv("SAVE_TENSOR_PATH", "/mnt/afs2/chenxuepan/work_dirs/save_tensors")
                for i in range(img.shape[0]):
                    sub = img[i].unsqueeze(0).contiguous()  # (1, 3, 576, 1024)
                    save_tensor(sub, self.save_dir, input_keys[i], "backbone")    
                save_tensor(img_feats_reshapeds[0], self.save_dir, "img_feats_0", "backbone",False)       
                save_tensor(img_feats_reshapeds[0], self.save_dir, "img_feats_1", "backbone",False)   
                save_tensor(img_feats_reshapeds[0], self.save_dir, "img_feats_2", "backbone",False)   
                save_tensor(img_feats_reshapeds[0], self.save_dir, "img_feats_3", "backbone",False)    
        # img_feats_reshapeds[-1].shape = torch.Size([2, 1, 7, 512, 17, 30])
        # np.save('align/backbone_result.npy', img_feats_reshaped[0].detach().cpu().numpy())
        return img_feats_reshapeds

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return [torch.nan_to_num(img_feat) for img_feat in img_feats]

    def obtain_history_memory(self,
                            img_metas=None,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_motions_3d=None,
                            gt_bboxes_2d=None,
                            gt_labels_2d=None,
                            gt_2d_to_3d_idx=None,
                            gt_cam_idx=None,
                            gt_only_2d_flag=None,
                            gt_bboxes_3d_cam=None,
                            gt_centers_2d=None,
                            gt_depths_cam=None,
                            gt_bboxes_ignore=None,
                            gt_plan_pos=None,
                            gt_obj_trajs=None,
                            map_gt_labels_3d=None,
                            map_gt_bboxes_3d=None,
                            gt_seg_mask=None,
                            gt_seg_offset=None,
                            gt_seg_type=None,
                            gt_seg_color=None,
                            gt_pv_seg_mask=None,
                            occ_voxel_semantics=None,
                            occ_voxel_instances=None,
                            occ_instance_class_ids=None,
                            occ_camera_masks=None,
                            data_tag=None,
                            pred_bbox3d_range_valid=None,
                            depth_map=None,
                            **data):
        losses = dict()
#随机性固定
        print("FWD_IN",
            "img", tensor_hash(data["img"]),
            "points", tensor_hash(data["points"]) if "points" in data else None,
            "gt_bboxes_3d", tensor_hash(gt_bboxes_3d),
            "map_gt_bboxes_3d", tensor_hash(map_gt_bboxes_3d))
#随机性固定
        T = data['img'].size(1)     # torch.Size([2, 1, 7, 3, 544, 960])
#随机性固定
        print("FWD_IN",
            "img", tensor_hash(data["img"]),
            "points", tensor_hash(data["points"]) if "points" in data else None,
            "gt_bboxes_3d", tensor_hash(gt_bboxes_3d),
            "map_gt_bboxes_3d", tensor_hash(map_gt_bboxes_3d))
#随机性固定
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses # 0
        if gt_motions_3d is None:
            gt_motions_3d = [None] * T
        if gt_bboxes_3d_cam is None:
            gt_bboxes_3d_cam = [None] * T
        if gt_plan_pos is None:
            gt_plan_pos = [None] * T
        if gt_obj_trajs is None:
            gt_obj_trajs = [None] * T
        if map_gt_labels_3d is None:
            map_gt_labels_3d = [None] * T
        if map_gt_bboxes_3d is None:
            map_gt_bboxes_3d = [None] * T
        if gt_seg_mask is None:
            gt_seg_mask = [None] * T
        if gt_seg_offset is None:
            gt_seg_offset = [None] * T
        if gt_seg_type is None:
            gt_seg_type = [None] * T
        if gt_seg_color is None:
            gt_seg_color = [None] * T
        if gt_pv_seg_mask is None:
            gt_pv_seg_mask = [None] * T
        if occ_voxel_semantics is None:
            occ_voxel_semantics = [None] * T
        if occ_voxel_instances is None:
            occ_voxel_instances = [None] * T
        if occ_instance_class_ids is None:
            occ_instance_class_ids = [None] * T
        if occ_camera_masks is None:
            occ_camera_masks = [None] * T
        if pred_bbox3d_range_valid is None:
            pred_bbox3d_range_valid = [None] * T
        if depth_map is None:
            depth_map = [None] * T
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key != 'img_feats':
                    if key == 'gt_lanelines':
                        data_t[key] = data[key]
                    elif key == 'map_gt_masks_3d':
                        data_t[key] = data[key]
                    elif key in ['map_gt_shifts_pts_list', 'map_gt_pts_types_list', 'map_gt_pts_colors_list']:
                        data_t[key] = data[key]
                    elif key in ['map_graph', 'onehot_category', 'points']:
                        data_t[key] = [sublist[i] for sublist in data[key]]
                    elif isinstance(data[key], list):
                        data_t[key] = data[key][i]
                    else:
                        data_t[key] = data[key][:, i] # 获得所有batch sample 某一时刻的样本

            data_t['img_feats'] = []
            for idx in range(len(data['img_feats'])): # for multi-layer feats output
                data_t['img_feats'].append(data['img_feats'][idx][:, i])
                # data_t['img_feats'].append(data['img_feats'][idx])
            # print(data_t['img_feats'][0].shape, 'data_t[img_feats][0].shape')
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True

            loss = self.forward_pts_train(img_metas[i],
                                          gt_bboxes_3d[i],
                                          gt_labels_3d[i],
                                          gt_motions_3d[i],
                                          gt_bboxes_2d[i],
                                          gt_labels_2d[i],
                                          gt_2d_to_3d_idx[i],
                                          gt_cam_idx[i],
                                          gt_only_2d_flag,
                                          gt_bboxes_3d_cam[i],
                                          gt_centers_2d[i],
                                          gt_depths_cam[i],
                                          gt_plan_pos[i],
                                          gt_obj_trajs[i],
                                          map_gt_labels_3d[i],
                                          map_gt_bboxes_3d[i],
                                          gt_seg_mask[i],
                                          gt_seg_offset[i],
                                          gt_seg_type[i],
                                          gt_seg_color[i],
                                          gt_pv_seg_mask[i],
                                          occ_voxel_semantics[i],
                                          occ_voxel_instances[i],
                                          occ_instance_class_ids[i],
                                          occ_camera_masks[i],
                                          data_tag,
                                          pred_bbox3d_range_valid[i],
                                          depth_map = depth_map[i],
                                          requires_grad=requires_grad,
                                          return_losses=return_losses,
                                          **data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]  # (544, 960)
        if isinstance(self.position_level, int):
            if os.getenv("DEPLOY") != 'True':
                bs, n = data['img_feats'][self.position_level].shape[:2]
                x = data['img_feats'][self.position_level].flatten(0, 1)
            else:
                bs, n = data['img_feats'].shape[:2]
                x = data['img_feats'].flatten(0, 1)
            location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)  # torch.Size([14, 17, 30, 2])
            return location
        elif isinstance(self.position_level, list):
            pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
            assert len(self.stride) == len(self.position_level)
            location_r = []
            for i, idx in enumerate(self.position_level):
                if idx >= len(data['img_feats']):
                    raise ValueError(f"idx: {idx}, len(data['img_feats']): {len(data['img_feats'])}")
                bs, n = data['img_feats'][idx].shape[:2]
                x = data['img_feats'][idx].flatten(0, 1)
                location = locations(x, self.stride[i], pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
                location_r.append(location)
            return location_r
        else:
            raise TypeError('self.position_level {} must be a int or list, but got {}'.format(
                                self.position_level, type(self.position_level)))

    def forward_roi_head(self, location, img_metas, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes': None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            bbox_dict = self.img_roi_head.get_bboxes(outs_roi, img_metas, **data)  # {'bbox_list': BN x (Mi, 4), 'bbox_score_list': BN x (Mi, 1)}
            outs_roi.update(bbox_dict)
            return outs_roi

    def forward_rpn_head(self, img_metas, **data):
        if not self.with_img_rpn:
            return None
        else:
            outs_rpn = self.img_rpn_head(img_metas, **data)
            return outs_rpn
    
    def data_process_for_pp(self, data, *args):
        batch_dict = {}
        for key in ['points', 'voxels', 'voxel_coords', 'voxel_num_points']:
            val = data[key]
            if key in ['voxels', 'voxel_num_points']:
                batch_dict[key] = torch.cat(val, dim=0) if isinstance(val, (list, tuple)) else val[0]
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):   # coor: (Ni, C)
                    idx_col = torch.full((coor.size(0), 1), i, dtype=coor.dtype, device=coor.device)
                    coor_pad = torch.cat([idx_col, coor], dim=1)  # (Ni, C+1)
                    coors.append(coor_pad)
                batch_dict[key] = torch.cat(coors, dim=0)

        for key in data:
            for sub_key in ['anchors', 'box_labels', 'reg_targets', 'reg_weights']:
                if sub_key in key:
                    batch_dict[key] = torch.stack(data[key], dim=0) if isinstance(data[key], (list, tuple)) else data[key]
        for arg in args:
            if arg in data:
                batch_dict[arg] = data[arg]
        return batch_dict

    def forward_lane2d_head(self, data):
        if self.lane2d_head is None:
            return None

        if 'lane2d_img_feats' in data:
            outs_lane2d_head = self.lane2d_head(data['lane2d_img_feats'][:,1,...]) # 注意这里只选用一个视角的图像特征来进行2D车道线的检测[bs,7,c,h,w] -> [bs,c,h,w]
        else:
            outs_lane2d_head = self.lane2d_head(data['img_feats'][0][:,1,...])
        data['lane_feature'] = outs_lane2d_head['lane_feature']
        return outs_lane2d_head

    def forward_pvb_head(self, location, img_metas,
                         gt_bboxes_3d, gt_labels_3d,
                         gt_bboxes_2d, gt_labels_2d,
                         gt_bboxes_3d_cam, gt_centers_2d,
                         gt_depths_cam, gt_cam_idx, cam_nums,
                         data,
                         pts_backbone_out=None,
                         ):
        if self.pts_bbox_head is None:
            return None, None, None, None
        def inner_run():
            outs_roi = self.forward_roi_head(location, img_metas, **data)
            outs_rpn = self.forward_rpn_head(img_metas, **data)
            topk_indexes = outs_roi['topk_indexes']
            nv2d_query_ref_points_pvb = None

            if outs_rpn is not None and gt_bboxes_3d_cam is not None:
                outs_rpn_ref_pinhole_pvb = outs_rpn + (None,)
                rpns_pinhole_pvb, _, _, _, centers3d_pinhole_pvb = self.img_rpn_head.get_reference_points(
                    *outs_rpn_ref_pinhole_pvb,
                    img_metas,
                    camera_type='pinhole',
                    cfg=None,
                    quant_restrict_ref_pts=True,
                    rescale=False,
                    all_cls_scores_2d=None,
                    **data
                )
                outs_rpn += (centers3d_pinhole_pvb,)
                pc_range_ = self.pts_bbox_head.bbox_coder.pc_range
                ref_pts = []
                for kk in range(len(img_metas)):
                    ret_pts_metakk = torch.cat(rpns_pinhole_pvb[kk*cam_nums:kk*cam_nums+cam_nums], dim=0)[:,:3]
                    ret_pts_metakk[:,0] = (ret_pts_metakk[:,0] - pc_range_[0]) / (pc_range_[3] - pc_range_[0])
                    ret_pts_metakk[:,1] = (ret_pts_metakk[:,1] - pc_range_[1]) / (pc_range_[4] - pc_range_[1])
                    ret_pts_metakk[:,2] = (ret_pts_metakk[:,2] - pc_range_[2]) / (pc_range_[5] - pc_range_[2])
                    ref_pts.append(ret_pts_metakk.unsqueeze(0))
                nv2d_query_ref_points_pvb = torch.cat(ref_pts, dim=0).clone().detach()
                # if self.quant_restrict_ref_pts and (torch.max(nv2d_query_ref_points_pvb) > 1 or torch.min(nv2d_query_ref_points_pvb) < 0):
                #     print('Unexpected_Point/Replace 0.5/Greater',torch.where(nv2d_query_ref_points_pvb > 1)[0].shape[0])
                #     print('Unexpected_Point/Replace 0.5/Smaller',torch.where(nv2d_query_ref_points_pvb < 0)[0].shape[0])
                #     nv2d_query_ref_points_pvb[torch.where(nv2d_query_ref_points_pvb > 1)] = 0.5
                #     nv2d_query_ref_points_pvb[torch.where(nv2d_query_ref_points_pvb < 0)] = 0.5                
                #     raise AssertionError('nv2d_query_ref_points_pvb beyond pc_range. Will cause quantization errors.')

            random_float = random.uniform(0, 1)
            if random_float < self.reset_probability:
                self.pts_bbox_head.reset_memory()
            
            if pts_backbone_out is not None:
                pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
                    pts_backbone_out['voxel_feats'], pts_backbone_out['voxel_coors'], pts_backbone_out['voxel_xyz'], pts_backbone_out['query_feats'],
                    pts_backbone_out['query_xyz'], pts_backbone_out['query_pred'], pts_backbone_out['query_cat'], len(gt_labels_3d))
            else:
                pts_feat, pts_pos, pts_query_feat, pts_query_center = None, None, None, None

            if "FarHead" in str(type(self.pts_bbox_head)):
                # if not self.with_img_roi_head:
                #     raise ValueError("FarHead must need use roi_head!")
                outs = self.pts_bbox_head(img_metas, outs_roi, pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                          pts_feat=pts_feat, pts_pos=pts_pos, pts_shape=None, **data)

            else:
                if self.pvb_position_encoder is None:
                    outs = self.pts_bbox_head(location, None, None, img_metas, topk_indexes, None, nv2d_query_ref_points_pvb, **data)
                else:
                    pvb_pe, pvb_cone = self.pvb_position_encoder(data, location, topk_indexes, img_metas)
                    outs = self.pts_bbox_head(None, pvb_pe, pvb_cone, img_metas, topk_indexes, None, nv2d_query_ref_points_pvb, **data)

            return outs, outs_roi, outs_rpn, random_float

        if self.fix_pvb_head:
            self.eval()
            with torch.no_grad():
                outs_spetr, outs_roi, outs_rpn, random_float = inner_run()
            self.train()
        else:
            outs_spetr, outs_roi, outs_rpn, random_float = inner_run()

        return outs_spetr, outs_roi, outs_rpn, random_float

    def forward_lane3d_head(self, location, topk_indexes, img_metas, data):
        if self.lane3d_head is None:
            return None

        # if self.lane3d_position_encoder is not None:
        #     lane3d_pe, lane3d_cone = self.lane3d_position_encoder(data, location, topk_indexes, img_metas)
        # else:
        #     lane3d_pe, lane3d_cone = None, None
        data = data.copy()  ####避免data被原地修改
        data['img_metas'] = img_metas
        outs_lane3d = self.lane3d_head(None, None, **data)
        #随机性固定
        TARGET_SAMPLE_ID = "s3://mapless-datasets-v2-laneline-t68/to_dongfeng/road-geometry/img/sdc3-adas-3/data_collection/gt_data/pilotGtRawParser/dongfengpro_vd_gt_collection/epai7-960/2025_12/2025_12_02/2025_12_02_18_54_56_AutoCollect_pilotGtRawParser/camera/center_camera_fov120 s3://mapless-datasets-v2-laneline-t68/to_dongfeng/road-geometry/img/sdc3-adas-3/data_collection/gt_data/pilotGtRawParser/dongfengpro_vd_gt_collection/epai7-960/2025_12/2025_12_02/2025_12_02_18_54_56_AutoCollect_pilotGtRawParser/camera/center_camera_fov30/1764671070304586452.jpg"
        meta0 = img_metas
        while isinstance(meta0, (list, tuple)):
            meta0 = meta0[0]

        sample_id = f'{meta0["scene_token"]} {meta0["filename"][0]}'
        self.hit_target = sample_id == TARGET_SAMPLE_ID
        self.debug_print = self.debug_train_iter < self.debug_train_n

        if self.hit_target and self.debug_print:
            print("SAMPLE_ID", sample_id)
        if self.hit_target and self.debug_print:
            print("LANE3D_KEYS", list(outs_lane3d.keys()))
            print("LANE3D_MAP_QUERIES", tensor_hash(outs_lane3d["map_queries"]) if "map_queries" in outs_lane3d else "NA")
            print("LANE3D_SEG", tensor_hash(outs_lane3d["seg"]) if "seg" in outs_lane3d else "NA")
            print("LANE3D_SEG_OFFSET", tensor_hash(outs_lane3d["seg_offset"]) if "seg_offset" in outs_lane3d else "NA")
            print("LANE3D_SEG_TYPE", tensor_hash(outs_lane3d["seg_type"]) if "seg_type" in outs_lane3d else "NA")
            print("LANE3D_SEG_COLOR", tensor_hash(outs_lane3d["seg_color"]) if "seg_color" in outs_lane3d else "NA")
        #随机性固定


        data['map_queries'] = outs_lane3d['map_queries']

        return outs_lane3d

    def forward_occ3d_head(self, img_metas, data):
        if self.occ3d_head is None:
            return None
        # if self.lane3d_head is not None:
        bev_feats = data.get('bev_embed', None)
        all_level_feats = data['img_feats']

        if len(bev_feats.shape)==3:
            B, _, C = bev_feats.shape
            # B, Dy, Dx, C
            bev_feats = bev_feats.view(B, self.lane3d_head.bev_h, self.lane3d_head.bev_w, C)
        bev_feats_for_occ = bev_feats.clone()
        if type(bev_feats_for_occ) in [list, tuple]:
            bev_feats_for_occ = bev_feats_for_occ[0]

        if self.lc_fusion:
            lidar_bev_feats = data.get('spatial_features_2d_lidar_gop_driving', None)
            print(f"lidar_bev_feats:{lidar_bev_feats}")
            bev_feats_for_occ = self.lc_fusion(bev_feats_for_occ, lidar_bev_feats)

        print()
        occ3d_outs = self.occ3d_head(bev_feats=bev_feats_for_occ, **data)
        return occ3d_outs
    # def forward_occ3d_head(self, img_metas, data):
    #     if self.occ3d_head is None:
    #         return None

    #     bev_feats = data.get('bev_embed', None)
    #     all_level_feats = data['img_feats']
    #     if len(bev_feats.shape)==3:
    #         B, _, C = bev_feats.shape
    #         bev_feats = bev_feats.view(B, self.lane3d_head.bev_h, self.lane3d_head.bev_w, C)
    #     bev_feats_for_occ = bev_feats.clone()
    #     if type(bev_feats_for_occ) in [list, tuple]:
    #         bev_feats_for_occ = bev_feats_for_occ[0]

    #     if self.lc_fusion:
    #         lidar_bev_feats = data.get('spatial_features_2d_lidar_gop_driving', None)
    #         bev_feats_for_occ = self.lc_fusion(bev_feats_for_occ, lidar_bev_feats)

    #     # fusion输出 NCHW (B,C,H,W)
    #     bev_feats_for_conv = bev_feats_for_occ.contiguous()
    #     # 关键：转为 BHWC (B,H,W,C) 给occ head内部卷积使用
    #     bev_feats_bhwc = bev_feats_for_conv.permute(0, 2, 3, 1).contiguous()
    #     occ3d_outs = self.occ3d_head(bev_feats=bev_feats_bhwc)
    #     return occ3d_outs
    def forward_planning_head(self, outs, random_float, data):
        if self.planning_head is None:
            return None

        # Optional reset memory
        if random_float < self.reset_probability:
            self.planning_head.reset_memory()
        memory = outs['memory']
        pos_embed = outs['pos_embed']
        object_queries = outs['object_queries']
        if self.fix_pvb_head:
            memory = memory.detach()
            object_queries = object_queries.detach()
            pos_embed = pos_embed.detach()
        outs_planning = self.planning_head(memory, pos_embed, object_queries, **data)

        return outs_planning
    
    def forward_gop_head(self, data, img_metas):

        if self.multimodal_gop is None:
            return
        ext_keys = ['pillar_features', 'spatial_features_2d_lidar_gop_driving', 'img_feats', 'data_tag', 'lidar2img', 'lidar_aug_matrix', 'extrinsics', 'intrinsics', 'cam_dist']
        batch_dict = self.data_process_for_pp(data, *ext_keys)
        out_dict = self.multimodal_gop(batch_dict)
        pts_feat_gop, pts_pos_gop, pts_query_feat_gop, pts_query_center_gop = None, None, None, None
        if hasattr(self, 'pts_query_generator_gop'):
            pts_feat_gop, pts_pos_gop, pts_query_feat_gop, pts_query_center_gop = self.pts_query_generator_gop(
                out_dict['voxel_feats'], out_dict['voxel_coors'], out_dict['voxel_xyz'], out_dict['query_feats'],
                out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], len(img_metas))
        if self.pts_bbox_head_gop is not None:
            if random.uniform(0, 1) < self.reset_probability:
                self.pts_bbox_head_gop.reset_memory()
            return  self.pts_bbox_head_gop(img_metas, {'topk_indexes': None}, pts_query_center=pts_query_center_gop, pts_query_feat=pts_query_feat_gop,
                                                    pts_feat=pts_feat_gop, pts_pos=pts_pos_gop, pts_shape=None, **data)
        return None

    def compute_losses(self, return_losses, params):
        if not return_losses:
            return None

        losses = {}

        img_metas = params['img_metas']
        cam_nums = params.get('cam_nums', 7)
        data = params['data']
        outs_lane2d_head = params['outs_lane2d_head']
        outs = params['outs_spetr']
        outs_gop = params['outs_spetr_gop']
        outs_roi = params['outs_roi']
        outs_rpn = params['outs_rpn']
        outs_lane3d = params.get('outs_lane3d', None)
        outs_occ3d = params.get('outs_occ3d', None)
        outs_planning = params.get('outs_planning', None)
        gt_bboxes_3d = params['gt_bboxes_3d']
        gt_labels_3d = params['gt_labels_3d']
        gt_motions_3d = params['gt_motions_3d']
        gt_bboxes_2d = params['gt_bboxes_2d']
        gt_labels_2d = params['gt_labels_2d']
        gt_only_2d_flag = params['gt_only_2d_flag']
        gt_2d_to_3d_idx = params['gt_2d_to_3d_idx']
        gt_cam_idx = params['gt_cam_idx']
        gt_bboxes_3d_cam = params['gt_bboxes_3d_cam']
        gt_centers_2d = params['gt_centers_2d']
        gt_depths_cam = params['gt_depths_cam']
        gt_obj_trajs = params['gt_obj_trajs']
        data_tag = params['data_tag']
        pred_bbox3d_range_valid = params['pred_bbox3d_range_valid']
        trajs_gts = params.get('trajs_gts', None)
        trajs_weights = params.get('trajs_weights', None)
        occ_voxel_semantics = params.get('occ_voxel_semantics', None)
        occ_voxel_instances = params.get('occ_voxel_instances', None)
        occ_instance_class_ids = params.get('occ_instance_class_ids', None)
        occ_camera_masks = params.get('occ_camera_masks', None)
        map_gt_bboxes_3d = params.get('map_gt_bboxes_3d', None)
        map_gt_labels_3d = params.get('map_gt_labels_3d', None)
        gt_seg_mask = params.get('gt_seg_mask', None)
        gt_seg_offset = params.get('gt_seg_offset', None)
        gt_seg_type = params.get('gt_seg_type', None)
        gt_seg_color = params.get('gt_seg_color', None)
        gt_pv_seg_mask = params.get('gt_pv_seg_mask', None)
        gt_plan_pos = params.get('gt_plan_pos', None)
        depth_map = params.get('depth_map', None)

        losses = {}

        if self.fix_pvb_head:
            if self.pred_frame_tf > 0 and self.pts_bbox_head is not None:
                self.eval()
                with torch.no_grad():
                    if "FarHead" in str(type(self.pts_bbox_head)):
                        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_motions_3d, outs, data_tag, pred_bbox3d_range_valid]
                    else:
                        loss_inputs = [img_metas, gt_bboxes_3d, gt_labels_3d, gt_motions_3d, gt_2d_to_3d_idx, gt_bboxes_2d, gt_cam_idx, gt_labels_2d, 
             gt_only_2d_flag, gt_obj_trajs, outs, data, data_tag, pred_bbox3d_range_valid]
                    losses = self.pts_bbox_head.loss(*loss_inputs)

                self.train()
            else:
                losses = {}
        else:
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes_2d, gt_labels_2d, gt_centers_2d, gt_depths_cam, gt_cam_idx, outs_roi, img_metas, data_tag]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)
            if self.with_img_rpn and outs_rpn is not None:
                loss2d_inputs_rpn = outs_rpn + (
                    gt_bboxes_2d, gt_labels_2d, gt_bboxes_3d_cam,
                    gt_labels_2d, gt_centers_2d, gt_depths_cam, gt_cam_idx
                ) + (cam_nums, img_metas)
                losses2d_rpn = self.img_rpn_head.loss(*loss2d_inputs_rpn)
                losses2d_rpn_new = {
                    "mono_" + key: value for key, value in losses2d_rpn.items() if 'loss' in key
                }
                losses.update(losses2d_rpn_new)
            if self.pts_bbox_head is not None:
                if "FarHead" in str(type(self.pts_bbox_head)):
                    loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_motions_3d, outs, data_tag, pred_bbox3d_range_valid]
                else:
                    loss_inputs = [img_metas, gt_bboxes_3d, gt_labels_3d, gt_motions_3d, gt_2d_to_3d_idx, gt_bboxes_2d, gt_cam_idx, gt_labels_2d, 
             gt_only_2d_flag, gt_obj_trajs, outs, data, data_tag, pred_bbox3d_range_valid]
                losses.update(self.pts_bbox_head.loss(*loss_inputs))

        trajs_gts = losses.pop('trajs_gts', None)
        trajs_weights = losses.pop('trajs_weights', None)

        if self.fix_pvb_head:
            if trajs_gts is not None:
                trajs_gts = trajs_gts.detach()
            if trajs_weights is not None:
                trajs_weights = trajs_weights.detach()

        if self.lane2d_head is not None:
            outs_lane2d_head = self.collate_lane2d_gt(outs_lane2d_head, data['gt_lanelines'])
            losses_lane2d = self.lane2d_head.get_loss(outs_lane2d_head)
            losses.update(losses_lane2d)

        if self.occ3d_head is not None:
            losses_occ = self.occ3d_head.loss(occ_voxel_semantics, occ_voxel_instances, occ_instance_class_ids, outs_occ3d, occ_camera_masks, data_tag=data_tag)
            losses.update(losses_occ)

        if self.lane3d_head is not None and not getattr(self.lane3d_head,'only_bev',False):
            losses_lane3d = self.lane3d_head.loss(map_gt_bboxes_3d, map_gt_labels_3d, outs_lane3d, data, depth_map=depth_map,
                                                    gt_seg_mask=gt_seg_mask, gt_seg_offset=gt_seg_offset,
                                                    gt_seg_type=gt_seg_type, gt_seg_color=gt_seg_color,
                                                    gt_pv_seg_mask=gt_pv_seg_mask, data_tag=data_tag)
            
            # import ipdb;ipdb.set_trace()
            if hasattr(self.lane3d_head,'k_one2many'):
                if self.lane3d_head.k_one2many > 0:
                    k_one2many = self.lane3d_head.k_one2many
                
                    multi_gt_bboxes_3d = copy.deepcopy(map_gt_bboxes_3d)
                    multi_gt_labels_3d = copy.deepcopy(map_gt_labels_3d)
                    new_multi_gt_labels_3d = []
                    for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
                        each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
                        each_gt_bboxes_3d.gt_labels = each_gt_bboxes_3d.gt_labels * k_one2many
                        each_gt_bboxes_3d.types_list = each_gt_bboxes_3d.types_list * k_one2many
                        each_gt_bboxes_3d.colors_list = each_gt_bboxes_3d.colors_list * k_one2many
                        each_gt_bboxes_3d.segpts_list = each_gt_bboxes_3d.segpts_list * k_one2many
                        new_multi_gt_labels_3d.append(each_gt_labels_3d.repeat(k_one2many))
                    multi_gt_labels_3d=tuple(new_multi_gt_labels_3d)
                    # import ipdb;ipdb.set_trace()
                    one2many_outs = outs_lane3d['one2many_outs']
                    loss_dict_one2many = self.lane3d_head.loss(multi_gt_bboxes_3d, multi_gt_labels_3d, one2many_outs, data, data_tag=data_tag)

                    lambda_one2many = self.lane3d_head.lambda_one2many
                    for key, value in loss_dict_one2many.items():
                        if key + "_one2many" in losses_lane3d.keys():
                            losses_lane3d[key + "_one2many"] += value * lambda_one2many
                        else:
                            losses_lane3d[key + "_one2many"] = value * lambda_one2many

            losses.update(losses_lane3d)

        if self.planning_head is not None:
            gt_keypoints = None
            light_signal = None
            if 'gt_lanelines' in data:
                gt_keypoints = [sample[0]['center_camera_fov120']['laneline']['key_points'] for sample in data['gt_lanelines']]
                light_signal = data.get('light_signal', torch.zeros((len(gt_keypoints), self.planning_head.light_frame), dtype=torch.int).cuda())

            extrinsics = data['extrinsics']
            intrinsics = data['intrinsics']
            if 'all_traj_preds' in outs:
                if not self.dynamic_collision or self.det_pred:
                    obj_trajs, obj_shapes = self.pts_bbox_head.get_obj_trajs(outs)
                else:
                    obj_trajs, obj_shapes = self.object2egocar(gt_obj_trajs, gt_bboxes_3d)
                loss_plan_inputs = [gt_plan_pos, outs_planning, obj_trajs, obj_shapes, gt_keypoints, extrinsics[:, 1], intrinsics[:, 1], light_signal, data_tag]
                losses_plan = self.planning_head.lossv2(*loss_plan_inputs)
            else:
                loss_plan_inputs = [gt_plan_pos, outs_planning, gt_bboxes_3d, gt_keypoints, extrinsics[:, 1], intrinsics[:, 1], light_signal, data_tag, trajs_gts, trajs_weights]
                losses_plan = self.planning_head.loss(*loss_plan_inputs)
            losses.update(losses_plan)

            if self.bev_constraint:
                bbox_results = self.outs2results(outs, outs_lane3d, outs_planning, map_gt_bboxes_3d, map_gt_labels_3d, img_metas, visualize=False)

                def dict2merge(bbox_results):
                    merge_dict = {}
                    for bbox in bbox_results:
                        for key, value in bbox.items():
                            if key not in merge_dict:
                                merge_dict[key] = [value]
                            else:
                                merge_dict[key].append(value)
                    return merge_dict

                merged_bbox_results = dict2merge(bbox_results)
                merged_bbox_results.update(light_signal=light_signal)
                merged_bbox_results.update(gt_plan_pos=gt_plan_pos)
                loss_plan_bev = self.planning_head.loss_plan_bev(img_metas=img_metas, **merged_bbox_results)
                losses.update(loss_plan_bev=loss_plan_bev)

            if os.getenv("DEBUG") == 'True':
                from mmcv.runner import get_dist_info
                rank, _ = get_dist_info()
                if rank == 0:
                    self.planning_head.visualization(img_metas, gt_plan_pos, outs_planning)
        if self.multimodal_gop is not None:
            for gop_task in self.multimodal_gop.multimodal_head.model_cfg.tasks:
                gop_task_name = f"dense_head_det_{gop_task}_gop_driving"
                _, gop_driving_loss = getattr(self.multimodal_gop.multimodal_head, gop_task_name).get_loss()
                losses.update(gop_driving_loss)
            if self.pts_bbox_head_gop is not None:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs_gop, data_tag, pred_bbox3d_range_valid]
                losses.update(self.pts_bbox_head_gop.loss(*loss_inputs))
        #随机性固定
        if self.debug_print:
            self.debug_train_iter += 1
        #随机性固定
        return losses

    def forward_pts_train(self,
                          img_metas,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_motions_3d,
                          gt_bboxes_2d,
                          gt_labels_2d,
                          gt_2d_to_3d_idx,
                          gt_cam_idx,
                          gt_only_2d_flag,
                          gt_bboxes_3d_cam,
                          gt_centers_2d,
                          gt_depths_cam,
                          gt_plan_pos,
                          gt_obj_trajs=None,
                          map_gt_labels_3d=None,
                          map_gt_bboxes_3d=None,
                          gt_seg_mask=None,
                          gt_seg_offset=None,
                          gt_seg_type=None,
                          gt_seg_color=None,
                          gt_pv_seg_mask=None,
                          occ_voxel_semantics=None,
                          occ_voxel_instances=None,
                          occ_instance_class_ids=None,
                          occ_camera_masks=None,
                          data_tag=None,
                          pred_bbox3d_range_valid=None,
                          requires_grad=True,
                          return_losses=False,
                          depth_map=None,
                          **data):
        """Forward function for point cloud branch.
        Returns:
            dict: Losses of each branch.
        """

        data['position_level'] = self.occ_levels
        location = self.prepare_location(img_metas, **data)
        cam_nums = data['img'].shape[1]
        data.update(data_tag=data_tag)

        if self.with_pts_backbone:
            if 'SingleStageFSDV2' in str(type(self.pts_backbone)) or 'FastPillar' in str(type(self.pts_backbone)):
                if self.pts_backbone.point_cloud_range != self.pts_bbox_head.pc_range.tolist():
                    new_gt_bboxes_3d = []
                    new_gt_labels_3d = []
                    bev_range = np.array(self.pts_backbone.point_cloud_range, dtype=np.float32)[[0, 1, 3, 4]]
                    for bboxes_3d, labels_3d in zip(gt_bboxes_3d, gt_labels_3d):
                        mask = bboxes_3d.in_range_bev(bev_range)
                        new_gt_bboxes_3d.append(bboxes_3d[mask])
                        new_gt_labels_3d.append(labels_3d[mask])
                    new_gt_bboxes_3d = tuple(new_gt_bboxes_3d)
                    new_gt_labels_3d = tuple(new_gt_labels_3d)
                    # if int(mask.sum()) == 0:
                    #     import ipdb;ipdb.set_trace()
                    out_dict = self.pts_backbone.forward_train(data['pts_feats'], img_metas, new_gt_bboxes_3d, new_gt_labels_3d)
                else:
                    out_dict = self.pts_backbone.forward_train(data['pts_feats'], img_metas, gt_bboxes_3d, gt_labels_3d)
                losses_pts = out_dict['losses']
                spatial_features_2d_lidar = out_dict['spatial_features_2d_lidar']
            else:
                batch_dict = self.data_process_for_pp(data, 'data_tag')
                batch_dict['fix_pts_backbone'] = self.fix_pts_backbone
                if self.fix_pts_backbone and self.pts_bbox_head:
                    self.eval()
                    out_dict = self.pts_backbone.forward_rpn(return_loss=True, **batch_dict)
                    #随机性固定
                    for name, value in out_dict.items():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                            self._debug_output_tensors[f'pts_backbone.{name}'] = value
                    #随机性固定
                    losses_pts = out_dict['losses']
                    self.train()
                elif self.fix_pts_backbone and not self.pts_bbox_head:
                    self.eval()
                    out_dict = self.pts_backbone.forward_rpn(return_loss=False, **batch_dict)
                    #随机性固定
                    for name, value in out_dict.items():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                            self._debug_output_tensors[f'pts_backbone.{name}'] = value
                    #随机性固定
                    losses_pts = {}
                    self.train()                    
                elif self.pts_bbox_head is None:
                    out_dict = self.pts_backbone.forward_rpn(return_loss=False, **batch_dict)
                    #随机性固定
                    for name, value in out_dict.items():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                            self._debug_output_tensors[f'pts_backbone.{name}'] = value
                    #随机性固定
                    losses_pts = {}
                else:
                    out_dict = self.pts_backbone.forward_rpn(return_loss=True, **batch_dict)
                    #随机性固定
                    for name, value in out_dict.items():
                        if torch.is_tensor(value) and value.requires_grad:
                            value.retain_grad()
                            self._debug_output_tensors[f'pts_backbone.{name}'] = value
                    #随机性固定
                    losses_pts = out_dict['losses']
                spatial_features_2d_lidar = out_dict['spatial_features_2d_lidar']
        else:
            out_dict = None
            spatial_features_2d_lidar = None

        if self.bev_encoder:
            mlvl_feats = [data['img_feats'][level] for level in self.occ_levels]
            bev_embed, bev_pos, depth = self.bev_encoder(
                mlvl_feats,
                lidar_feat=spatial_features_2d_lidar,
                data=data,
                intrinsics = data['intrinsics'],
                extrinsics = data['extrinsics'],
                lidar2img = data['lidar2img'],
                img_metas = img_metas,
                )
            #随机性固定
            if bev_embed.requires_grad:
                bev_embed.retain_grad()
                self._debug_output_tensors['bev_embed'] = bev_embed
            if self.hit_target and self.debug_print:
                print("BEV_EMBED", tensor_hash(bev_embed))
            #随机性固定
            data['bev_embed'] = bev_embed

        #前 N 帧先固定模型参数（不训练，减少计算开销或避免初始帧数据扰动模型）
        if not requires_grad:#不计算梯度，参数固定不变（冻结，不参与训练）
            self.eval()
            with torch.no_grad():
                location_new = location.clone() if isinstance(location, torch.Tensor) else [t.clone() for t in location]
                outs_spetr = self.pts_bbox_head(location_new, img_metas, None, **data)
            self.train()
        #从第 N+1 帧开始再让参数参与训练更新
        else:#计算梯度，参数会在反向传播时被优化器更新（参与训练）
            outs_lane2d_head = self.forward_lane2d_head(data)
            location_new = location.clone() if isinstance(location, torch.Tensor) else [t.clone() for t in location]
            outs_spetr, outs_roi, outs_rpn, random_float = self.forward_pvb_head(location_new, img_metas,
                                                                                 gt_bboxes_3d, gt_labels_3d,
                                                                                 gt_bboxes_2d, gt_labels_2d,
                                                                                 gt_bboxes_3d_cam, gt_centers_2d,
                                                                                 gt_depths_cam, gt_cam_idx, cam_nums,
                                                                                 data,
                                                                                 out_dict
                                                                                 )
            data.update(spatial_features_2d_lidar_gop_driving=spatial_features_2d_lidar)
            outs_spetr_gop = self.forward_gop_head(data, img_metas)
            location_new = location.clone() if isinstance(location, torch.Tensor) else [t.clone() for t in location]
            outs_lane3d = self.forward_lane3d_head(location_new, None, img_metas, data)
            #随机性固定
            for name, value in outs_lane3d.items():
                if torch.is_tensor(value) and value.requires_grad:
                    value.retain_grad()
                    self._debug_output_tensors[f'lane3d.{name}'] = value
            if self.hit_target and self.debug_print:
                print("LANE3D_KEYS", list(outs_lane3d.keys()))
                print("LANE3D_MAP_QUERIES", tensor_hash(outs_lane3d["map_queries"]) if "map_queries" in outs_lane3d else None)
                print("LANE3D_SEG", tensor_hash(outs_lane3d["seg"]) if "seg" in outs_lane3d else None)
                print("LANE3D_SEG_OFFSET", tensor_hash(outs_lane3d["seg_offset"]) if "seg_offset" in outs_lane3d else None)
                print("LANE3D_SEG_TYPE", tensor_hash(outs_lane3d["seg_type"]) if "seg_type" in outs_lane3d else None)
                print("LANE3D_SEG_COLOR", tensor_hash(outs_lane3d["seg_color"]) if "seg_color" in outs_lane3d else None)
                print("LANE3D_OUT",
                tensor_hash(outs_lane3d["bev_embed"]) if "bev_embed" in outs_lane3d else None)
            #随机性固定
            if outs_lane3d is not None and 'bev_embed' in outs_lane3d:
                data['bev_embed'] = outs_lane3d['bev_embed']

            outs_occ3d = self.forward_occ3d_head(img_metas, data)
            outs_planning = self.forward_planning_head(outs_spetr, random_float, data)
        loss_input_params = {
            'img_metas': img_metas,
            'cam_nums': cam_nums,
            'data': data,
            'outs_lane2d_head': outs_lane2d_head,
            'outs_roi': outs_roi,
            "outs_rpn": outs_rpn,
            'outs_spetr': outs_spetr,
            'outs_spetr_gop': outs_spetr_gop,
            'outs_lane3d': outs_lane3d,
            'outs_occ3d': outs_occ3d,
            'outs_planning': outs_planning,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'gt_motions_3d': gt_motions_3d,
            'gt_bboxes_2d': gt_bboxes_2d,
            'gt_labels_2d': gt_labels_2d,
            'gt_2d_to_3d_idx': gt_2d_to_3d_idx,
            'gt_cam_idx': gt_cam_idx,
            'gt_only_2d_flag': gt_only_2d_flag,
            'gt_bboxes_3d_cam': gt_bboxes_3d_cam,
            'gt_centers_2d': gt_centers_2d,
            'gt_depths_cam': gt_depths_cam,
            'gt_obj_trajs': gt_obj_trajs,
            'occ_voxel_semantics': occ_voxel_semantics,
            'occ_voxel_instances': occ_voxel_instances,
            'occ_instance_class_ids': occ_instance_class_ids,
            'occ_camera_masks': occ_camera_masks,
            'map_gt_bboxes_3d': map_gt_bboxes_3d,
            'map_gt_labels_3d': map_gt_labels_3d,
            'gt_seg_mask': gt_seg_mask,
            'gt_seg_offset': gt_seg_offset, 
            'gt_seg_type': gt_seg_type, 
            'gt_seg_color': gt_seg_color,
            'gt_pv_seg_mask': gt_pv_seg_mask,
            'gt_plan_pos': gt_plan_pos,
            'data_tag': data_tag,
            'pred_bbox3d_range_valid': pred_bbox3d_range_valid,
            'depth_map': depth_map,
        }

        losses = self.compute_losses(return_losses, loss_input_params)
        if self.with_pts_backbone:
            losses.update(losses_pts)

        return losses
    
    def outs2results(self, outs, lane3d_outs, plan_pos, map_gt_bboxes_3d, map_gt_labels_3d, img_metas, visualize=False):
        '''
        outs -- outs from pts_bbox_head
        lane3d_outs -- outs from lane3d_head. 
        plan_pos -- outs from mmplanning_head.    clone() it 
        map_gt_bboxes_3d -- gt_points from gt .   deep_copy() it
        map_gt_labels_3d -- gt_label of the 3d laneline ,just for filtering the cross walk and stop line
        this transform should not update by loss grad backward --- with torch.no_grad
        '''
        ########################prediction#########################
        # preprocess for plan_pos

        plan_pos = plan_pos['plan_coords_preds'].clone()
        B, _ , N, _ = plan_pos.shape # 这里的N为1
        plan_pos = plan_pos.reshape(B, -1, 2, N)
        if self.pred_delta:
            for i in range(plan_pos.shape[1] - 1):
                plan_pos[:, i+1] = plan_pos[:, i] + plan_pos[:, i+1]
        else:
            for i in range(plan_pos.shape[1]):
                plan_pos[:, i] *= (i + 1)
        if self.pred_frame_tf > 0:
            obj_trajs_tf = plan_pos[..., 1:].clone()
            obj_trajs_tf = obj_trajs_tf.permute(0, 3, 1, 2).contiguous()
        plan_pos = plan_pos[..., 0:1]
        # postprocess for det_outs
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = []
        for i, (bboxes, scores, labels, mask) in enumerate(bbox_list):
            bbox_result = self.bbox3d2result(bboxes, scores, labels)
            if self.pred_frame_tf > 0:
                obj_trajs_tf_mask = obj_trajs_tf[i][mask]
                bbox_result['obj_trajs_tf'] = obj_trajs_tf_mask.cpu()
                for j in range(len(bbox_result['obj_trajs_tf'])):
                    obj_pos = bbox_result['boxes_3d'].tensor[j].numpy()[:2]
                    obj_yaw = bbox_result['boxes_3d'].tensor[j].numpy()[6]
                    rot = np.array([[math.cos(obj_yaw), math.sin(obj_yaw)],
                                    [-math.sin(obj_yaw), math.cos(obj_yaw)]])
                    bbox_result['obj_trajs_tf'][j] = bbox_result['obj_trajs_tf'][j] @ rot + obj_pos
            bbox_results.append(bbox_result)

        if 'all_traj_preds' in outs:
            obj_trajs, obj_shapes = self.pts_bbox_head.get_obj_trajs(outs)
            for i in range(len(obj_trajs)):
                bbox_results[i].update(obj_trajs=obj_trajs[i].cpu())
        
        if self.lane3d_head is not None:
            map_bbox_list = self.lane3d_head.get_bboxes(lane3d_outs, img_metas)
            # assert len(map_bbox_list) == B
            for i in range(len(map_bbox_list)):
                map_scores, map_labels, map_pts, map_pts_types, map_pts_colors = map_bbox_list[i][:5]
                map_pts_uncertain = None
                if len(map_bbox_list[i]) > 5:
                    map_pts_uncertain = map_bbox_list[i][5]
                map_bbox_result = self.map_pred2result(
                    map_scores, map_labels, map_pts, map_pts_types, map_pts_colors, pts_uncertain=map_pts_uncertain)
                bbox_results[i].update(map_bbox_result)
        ############################gt#############################
        # process for map_gt_3d_pts
        # import pdb;pdb.set_trace()
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
        # NOTE: filter cross walks and stop_line from all the 3dlaneline according to the label
        for i in range(len(map_gt_labels_3d)):
            sample_lane3d_lines = map_gt_vecs_list[i].instance_list # a list of LiDARInstanceLines objects
            sample_lane3d_labels = map_gt_labels_3d[i] # a tensor of labels
            sample_lane3d_lines = [line for idx, line in enumerate(sample_lane3d_lines) if sample_lane3d_labels[idx] < 2] # filtering label >= 2 lane3d_line
            if len(sample_lane3d_lines) == 0:
                # import pdb;pdb.set_trace()
                point_cloud_range=[-100.0, -50.0, -5.0, 100.0, 50.0, 3.0]
                fake_line_pt = [point_cloud_range[0], point_cloud_range[1]]
                from shapely.geometry import LineString
                sample_lane3d_lines= [LineString([fake_line_pt, fake_line_pt])]
            map_gt_vecs_list[i].instance_list = sample_lane3d_lines #  using the filtered laneline set to replace the origin 
            # TODO: process filtering the corresponding the label as well
    
        for i in range(len(map_gt_vecs_list)):
            gt_bboxes = map_gt_vecs_list[i]
            bbox_results[i].update(map_gt_pts_3d=gt_bboxes.shift_fixed_num_sampled_points_v2.cpu()) # [num_instance, num_shifts, 20, 2]
        # if rpn_proposals is not None:
        #     assert len(bbox_results) == len(rpn_proposals[0]) == len(rpn_proposals[1]) == len(rpn_proposals[2])
        #     for i in range(len(bbox_results)):
        #         bbox_results[i].update(reference_points=rpn_proposals[0][i].cpu())
        #         bbox_results[i].update(top_proposals_bboxes_2d=rpn_proposals[1][i].cpu())
        #         bbox_results[i].update(top_proposals_scores=rpn_proposals[2][i].cpu())
        
        if self.planning_head is not None:
            assert len(bbox_results) == B
            for i in range(len(bbox_results)):
                bbox_results[i].update(plan_pos=plan_pos[i].squeeze())
        # if self.lane2d_head is not None:
        #     # 多batch infer后处理暂不支持
        #     assert len(bbox_results) == 1
        #     bbox_results[0].update(lane_points=outs_lane2d_postprocess)
        #########Visualization############
        if visualize:
            for i in range(len(bbox_results)):
                self.intermediate_visual(img_metas[i], **bbox_results[i])
        return bbox_results
    
    '''
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))
    '''
    def intermediate_visual(self, img_meta, plan_pos=None, obj_trajs=None, boxes_3d=None, map_pts_3d=None, map_gt_pts_3d=None,**kwargs):
        import numpy as np
        import mmcv
        import cv2

        def transform_to_bv_image(point, bv_img_w, bv_img_h, bv_resolution_y=8, bv_resolution_x=8):
            # return (-int(point[1] * self.bv_resolution_y - self.bv_img_w / 2),
            #         -int(point[0] * self.bv_resolution_x - 7 * self.bv_img_h / 10))
            return (-int(point[1] * bv_resolution_y - bv_img_w / 2),
                    -int(point[0] * bv_resolution_x - 100 * bv_img_h / 180))

        ## get 120 image, hard code!
        from path_mapping import PATH_MAPPING
        path_mapping = PATH_MAPPING
        file_client = mmcv.FileClient(
            backend='petrel', conf_path='./petreloss.conf', path_mapping=path_mapping)
        # draw the self plan pos
        image_file = img_meta['filename']
        image_path = image_file[1] # center_camera_fov120
        if not file_client.exists(image_path):
            image_path = os.path.splitext(image_path)[0]+'.jpg'
        image = mmcv.imfrombytes(file_client.get(image_path))
        image_h = 1080
        image_w = 1920
        image = cv2.resize(image, (image_w, image_h))

        ## draw bev
        bv_img_h = image_h
        bv_img_w = image_h
        image_bv = np.ones([bv_img_h, bv_img_w, 3], dtype=np.uint8) * 0
        ## pred pos
        for point in plan_pos:
            cv2.circle(image_bv, transform_to_bv_image([point[0], point[1]], bv_img_w, bv_img_h),
                    5, (0, 0, 255), -1)
        cv2.circle(image_bv, transform_to_bv_image([plan_pos[0][0], plan_pos[0][1]], bv_img_w, bv_img_h),
                    8, (0, 255, 0), -1)
        for obj_traj in obj_trajs:
            for point in obj_traj:
                cv2.circle(image_bv, transform_to_bv_image([point[0], point[1]], bv_img_w, bv_img_h),
                    5, (255, 0, 0), -1)
        # draw object bbox
        # draw object trajs
        # obj_trajs.reshape(-1,2).squeeze()
        # for i in range(obj_trajs.shape[0]-1):
        #     obj_trajs[i+1] += obj_trajs[i] 
        # draw 3d prediction laneline 
        for polygon in map_pts_3d:
            n = polygon.shape[0]
            for i in range(n-1):
                pt1 = transform_to_bv_image([polygon[i][0], polygon[i][1]], bv_img_w, bv_img_h)
                pt2 = transform_to_bv_image([polygon[i+1][0], polygon[i+1][1]], bv_img_w, bv_img_h)
                cv2.line(image_bv, pt1, pt2, (255, 255, 255), 5)
        # draw 3d gt laneline
        lane3d_gt_pts = map_gt_pts_3d.reshape(map_gt_pts_3d.shape[0],-1, 2)
        for polygon in lane3d_gt_pts:
            for point in polygon:
                cv2.circle(image_bv, transform_to_bv_image([point[0], point[1]], bv_img_w, bv_img_h), 5, (255, 255, 0), -1)
            # for i in range(n-1):
            #     pt1 = transform_to_bv_image([polygon[i][0], polygon[i][1]], bv_img_w, bv_img_h)
            #     pt2 = transform_to_bv_image([polygon[i+1][0], polygon[i+1][1]], bv_img_w, bv_img_h)
            #     cv2.line(image_bv, pt1, pt2, (255, 255, 0), 5)

        ## visualize for the selected laneline3d
        def select_laneline(plan_pos, lane3d_gt_pts):
            #根据起点进行选择相邻近车道线
            start_point = plan_pos[0].detach().cpu().numpy()
            select_left_idx = -1
            dis_left = 1e6
            select_right_idx = -1
            dis_right = -1e6            
            for i, laneline in enumerate(lane3d_gt_pts):
                lane_x = laneline[:,0]
                lane_y = laneline[:,1]
                interpolator = interp1d(lane_x, lane_y, bounds_error=False, fill_value=1e6)
                near_x = interpolator(start_point[0])
                cv2.circle(image_bv, transform_to_bv_image([start_point[0], near_x], bv_img_w, bv_img_h), 8, (10,215,255), -1)
                dis = start_point[1] - near_x
                if dis > 0 and dis < dis_left:
                    dis_left = dis
                    select_left_idx = i
                elif dis < 0 and dis > dis_right:
                    dis_right = dis
                    select_right_idx = i
            lane_left = lane3d_gt_pts[select_left_idx]
            lane_right = lane3d_gt_pts[select_right_idx]
            return lane_left, lane_right

        laneline_left, laneline_right = select_laneline(plan_pos, lane3d_gt_pts)
        for point in laneline_left:
            cv2.circle(image_bv, transform_to_bv_image([point[0], point[1]], bv_img_w, bv_img_h), 5, (255, 0, 255), -1)
        for point in laneline_right:
            cv2.circle(image_bv, transform_to_bv_image([point[0], point[1]], bv_img_w, bv_img_h), 5, (0, 255, 255), -1)

        image_full = cv2.hconcat([image, image_bv])
        cv2.imwrite("/mnt/lustrenew/shilong/store/{}".format(image_path.replace('/','-')), image_full)

        return

    def intermediate_visual_gt(self, imgs, img_metas, intrinsics, extrinsics,  map_gts_pts_3d=None,  map_gts_labels_3d=None, gts_bboxes_3d=None, occ_voxel_semantics=None):
        import numpy as np
        import mmcv
        import cv2
        from shapely import LineString
        from tools.misc.senseauto_bev_tools import lidar2image, interpolate_line, project_to_camera_lane, pts_transfer

        def draw_dict_top_right(img, info_dict, font_scale=0.6, font_thickness=1, padding=10, line_spacing=5):
            img_copy = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 获取最大宽度与总高度
            max_text_width = 0
            total_height = 0
            text_sizes = []

            for key, value in info_dict.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"

                (w, h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_sizes.append(((w, h), text))
                max_text_width = max(max_text_width, w)
                total_height += h + line_spacing

            # 起始点 (右上角留padding)
            x_start = img.shape[1] - max_text_width - 2 * padding
            y_start = padding

            # 画背景框（可选）
            cv2.rectangle(img_copy, 
                        (x_start - padding, y_start - padding), 
                        (x_start + max_text_width + padding, y_start + total_height), 
                        (0, 0, 0), 
                        thickness=-1)  # 黑色背景

            # 输出文字
            y = y_start
            for (size, text) in text_sizes:
                (w, h) = size
                cv2.putText(img_copy, text, (x_start, y + h), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                y += h + line_spacing

            return img_copy

        def transform_to_bv_image(point, bv_img_w, bv_img_h, bv_resolution_y=8, bv_resolution_x=8):
            return (-int(point[1] * bv_resolution_y - bv_img_w / 2),
                    -int(point[0] * bv_resolution_x - 100 * bv_img_h / 180))

        def draw_rotated_rectangle(image, center, theta, width, height, color):
            bv_img_h = 544
            bv_img_w = 960
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.matrix('{c} {0}; {1} {c}'.format(-s, s, c=c))
            p1 = [+width / 2, +height / 2]
            p2 = [-width / 2, +height / 2]
            p3 = [-width / 2, -height / 2]
            p4 = [+width / 2, -height / 2]
            p1_new = np.dot(p1, R) + center
            p2_new = np.dot(p2, R) + center
            p3_new = np.dot(p3, R) + center
            p4_new = np.dot(p4, R) + center

            # draw roated bbox
            # color = self.color_map.get(mode, (0, 0, 255))
            line_width = 2
            cv2.line(image,
                    transform_to_bv_image([p1_new[0, 0], p1_new[0, 1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p2_new[0, 0], p2_new[0, 1]], bv_img_w, bv_img_h),
                    color, line_width)
            cv2.line(image,
                    transform_to_bv_image([p2_new[0, 0], p2_new[0, 1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p3_new[0, 0], p3_new[0, 1]], bv_img_w, bv_img_h),
                    color, line_width)
            cv2.line(image,
                    transform_to_bv_image([p3_new[0, 0], p3_new[0, 1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p4_new[0, 0], p4_new[0, 1]], bv_img_w, bv_img_h),
                    color, line_width)
            cv2.line(image,
                    transform_to_bv_image([p4_new[0, 0], p4_new[0, 1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p1_new[0, 0], p1_new[0, 1]], bv_img_w, bv_img_h),
                    color, line_width)
            cv2.line(image, transform_to_bv_image([center[0], center[1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p1_new[0, 0], p1_new[0, 1]], bv_img_w, bv_img_h),
                    color, 3)
            cv2.line(image, transform_to_bv_image([center[0], center[1]], bv_img_w, bv_img_h),
                    transform_to_bv_image([p4_new[0, 0], p4_new[0, 1]], bv_img_w, bv_img_h),
                    color, 3)

            return image
        
        def draw_rect3d_on_img(image,
                           rect_corners,
                           color=(0, 255, 0),
                           thickness=1,
                           image_w=960,
                           image_h=544):
            """Plot the boundary lines of 3D rectangular on 2D images.

            Args:
                img (numpy.array): The numpy array of image.
                num_rects (int): Number of 3D rectangulars.
                rect_corners (numpy.array): Coordinates of the corners of 3D
                    rectangulars. Should be in the shape of [num_rect, 8, 2].
                color (tuple[int], optional): The color to draw bboxes.
                    Default: (0, 255, 0).
                thickness (int, optional): The thickness of bboxes. Default: 1.
            """

            if len(rect_corners) != 8:
                # draw outsider
                bbox_lu, bbox_rd = [image_w, image_h], [0, 0]
                for i in range(len(rect_corners)):
                    bbox_lu[0] = max(min(bbox_lu[0], rect_corners[i][0]), 0)
                    bbox_lu[1] = max(min(bbox_lu[1], rect_corners[i][1]), 0)
                    bbox_rd[0] = min(
                        max(bbox_rd[0], rect_corners[i][0]), image_w)
                    bbox_rd[1] = min(
                        max(bbox_rd[1], rect_corners[i][1]), image_h)

                if np.any((np.array(bbox_rd) - np.array(bbox_lu)) <= 0):
                    return image
                cv2.rectangle(image, (int(bbox_lu[0]), int(bbox_lu[1])), (int(
                    bbox_rd[0]), int(bbox_rd[1])), color, 2)
                return image

            #
            line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                            (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
            # heading_line_indices = ((4, 6), (5, 7))
            heading_line_indices = ((0, 5), (1, 4))

            corners = rect_corners.astype(np.int)
            for start, end in line_indices:
                cv2.line(image, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)
            for start, end in heading_line_indices:
                cv2.line(image, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)

            return image

        def to_bev_vis_mask(voxels, mask_camera,
            FREE_LABEL=2,GROUND_LABEL=1,
            voxel_size=0.4,
            start_idx=0, end_idx=88,
            TOGOROUD_LABEL=1,TO_OCC_LABEL=0,
                ):
                bev_grids = np.ones(voxels.shape[:2]) * FREE_LABEL
                voxels_change = voxels[..., start_idx:end_idx].copy()

                occ_free = np.logical_and.reduce((voxels_change == FREE_LABEL), axis=2)
                occ_others = (np.logical_or.reduce((voxels_change != GROUND_LABEL) & (voxels_change != FREE_LABEL), axis=2) & (~occ_free))
                occ_grond = (np.logical_and.reduce((voxels_change == GROUND_LABEL) | (voxels_change == FREE_LABEL), axis=2) & (~occ_free))
                bev_grids[occ_others] = TO_OCC_LABEL
                bev_grids[occ_grond] = TOGOROUD_LABEL
                return bev_grids

        label_color_mp = {
            0: (255, 255, 255),
            1: (0, 255, 9),
            2: (251, 255, 0),
            3: (0, 0, 255),
            4: (0, 255, 255),
        }

        od_color = (0, 0, 255)

        occ_color = np.array([[127,80,200, 100], [191, 207, 0, 100], [255,255,255,100]])

        pt_type_mapping = {
            idx: ('dashed_lane' if idx in(1, 5, 7, 9) else 'solid_lane') for idx in range(28)
        }

        ## get 120 image, hard code!
        from path_mapping import PATH_MAPPING
        path_mapping = PATH_MAPPING
        file_client = mmcv.FileClient(
            backend='petrel', conf_path='./petreloss.conf', path_mapping=path_mapping)

        for idx in range(len(img_metas[0])):
            img_meta = img_metas[0][idx]
            map_gt_pts_3d = map_gts_pts_3d[0][idx]
            map_gt_labels_3d = map_gts_labels_3d[0][idx]
            gt_bboxes_3d = gts_bboxes_3d[0][idx].tensor.numpy()
            voxel_label = occ_voxel_semantics[0][idx].cpu().numpy()
            image_file = img_meta['filename']
            image_path = image_file[1] # center_camera_fov120
            intrinsic = intrinsics[idx][0][1]
            extrinsic = extrinsics[idx][0][1]
            if not file_client.exists(image_path):
                image_path = os.path.splitext(image_path)[0]+'.jpg'
            image = imgs[idx][0][1].cpu().numpy()
            aug_info = img_meta['pos_aug_info'][1] if 'pos_aug_info' in img_meta else {}
            mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
            std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
            image = ((image * std) + mean)[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.uint8).copy()
            # image = mmcv.imfrombytes(file_client.get(image_path))
            # image_h = 1080
            # image_w = 1920
            # image = cv2.resize(image, (image_w, image_h))
            image_h, image_w = image.shape[:-1]

            ## draw bev
            bv_img_h = image_h
            bv_img_w = image_w
            image_bv = np.ones([bv_img_h, bv_img_w, 3], dtype=np.uint8) * 0
            # draw ego car
            cv2.rectangle(image_bv, transform_to_bv_image([2.4125, 0.965], bv_img_w, bv_img_h),
                        transform_to_bv_image([-2.4125, -0.965], bv_img_w, bv_img_h), (0, 0, 255), 2)
            shifted_pts_and_attrs_list = map_gt_pts_3d.shift_fixed_num_sampled_points_v6
            # draw 3d gt laneline
            shifted_pts_and_attrs_list = [tensor[:, 1] for tensor in shifted_pts_and_attrs_list]
            all_pts_list = []
            all_color_list = []
            for polygon, label, pts_type, pts_color in zip(
                    torch.unbind(shifted_pts_and_attrs_list[0], dim=0),
                    torch.unbind(map_gt_labels_3d, dim=0),
                    torch.unbind(shifted_pts_and_attrs_list[1], dim=0),
                    torch.unbind(shifted_pts_and_attrs_list[2], dim=0)):
                label = label.cpu().item()
                color = label_color_mp[label]

                polygon = polygon.unbind(dim=0)
                pts_type = pts_type.unbind(dim=0)
                pts_color = pts_color.unbind(dim=0)

                sampled_points = [point.tolist() for point in polygon]
                sampled_types = [pt_type.item() for pt_type in pts_type]
                sampled_colors = [pt_color.item() for pt_color in pts_color]
                n = len(sampled_points)
                for i in range(n - 1):
                    if sampled_points[i][0] == -10000 or sampled_points[i + 1][0] == -10000:
                        continue
                    pt1 = transform_to_bv_image([sampled_points[i][0], sampled_points[i][1]], bv_img_w, bv_img_h)
                    pt2 = transform_to_bv_image([sampled_points[i + 1][0], sampled_points[i + 1][1]], bv_img_w, bv_img_h)
                    pt_type = sampled_types[i]
                    pt_color = sampled_colors[i]
                    if label == 0 and pt_color == 1:
                        color = (0, 255, 255) #yellow
                    mid_pt = tuple(int((a + b) / 2) for a, b in zip(pt1, pt2))
                    if pt_type_mapping[pt_type] == 'dashed_lane':
                        cv2.line(image_bv, pt1, mid_pt, color, 5)
                    else:
                        cv2.line(image_bv, pt1, mid_pt, color, 5)
                        cv2.line(image_bv, mid_pt, pt2, color, 5)
                    # cv2.circle(image_bv, pt1, 5, color, -1)

                cv2.circle(image_bv, transform_to_bv_image([sampled_points[0][0], sampled_points[0][1]], bv_img_w, bv_img_h), 8, color, -1)
                cv2.circle(image_bv, transform_to_bv_image([sampled_points[-1][0], sampled_points[-1][1]], bv_img_w, bv_img_h), 8, color, -1)

                bv_pts = interpolate_line(LineString(sampled_points), 50)
                trans_pts = pts_transfer(bv_pts)
                img_pts = project_to_camera_lane(trans_pts, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy())
                for j in range(img_pts.shape[0]-1):
                    x1 = int(img_pts[j][0])
                    y1 = int(img_pts[j][1])
                    x2 = int(img_pts[j+1][0])
                    y2 = int(img_pts[j+1][1])
                    try:
                        cv2.line(image, (x1, y1), (x2, y2), color, 3)
                    except:
                        pass
                all_pts_list.append(trans_pts)
                all_color_list.append(color)
            image = draw_dict_top_right(image, aug_info)
            # draw 3d gt object
            for single_bbox in gt_bboxes_3d:
                x, y, z, l, w, h, yaw, _, _ = single_bbox
                image_bv = draw_rotated_rectangle(image_bv, np.array([x, y]), yaw, l, w, od_color)
                z = z + 0.5 * h
                bbox_3d = [x, y, z, l, w, h, yaw]
                bbox_3d_image = lidar2image(
                        bbox_3d, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy(), image_w, image_h)
                if bbox_3d_image is not None:
                    draw_rect3d_on_img(image, bbox_3d_image, (61, 102, 255))

            # draw 3d gt occ
            image_bv_occ = np.ones([bv_img_h, bv_img_w, 3], dtype=np.uint8) * 0
             # draw ego car
            cv2.rectangle(image_bv_occ, transform_to_bv_image([2.4125, 0.965], bv_img_w, bv_img_h),
                        transform_to_bv_image([-2.4125, -0.965], bv_img_w, bv_img_h), (0, 0, 255), 2)
            camera_mask = None # if 'mask_camera' in data_pred else None
            if camera_mask is None:
                camera_mask = np.ones(voxel_label.shape, dtype=bool)
            # import pdb; pdb.set_trace()
            bev_gt = to_bev_vis_mask(voxel_label, camera_mask)
            bev_gt = cv2.resize(bev_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            bev_gt_col = np.zeros([bev_gt.shape[0], bev_gt.shape[1], 3])
            bev_gt_col[bev_gt==0,:] = occ_color[0][:3] # others
            bev_gt_col[bev_gt==1,:] = occ_color[1][:3] # ground
            bev_gt_col[bev_gt==2,:] = occ_color[2][:3] # free
            # TODO add cam mask
            image_bv_occ[:] = cv2.rotate(bev_gt_col, cv2.ROTATE_180)

            image_full = cv2.hconcat([image, image_bv, image_bv_occ])
            img_list = []
            # hard code for show all img
            for j in [0,2,3]:
                image = imgs[idx][0][j].cpu().numpy()
                aug_info = img_meta['pos_aug_info'][j] if 'pos_aug_info' in img_meta else {}
                intrinsic = intrinsics[idx][0][j]
                extrinsic = extrinsics[idx][0][j]
                mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
                std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
                image = ((image * std) + mean)[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.uint8).copy()
                for trans_pts,color in zip(all_pts_list,all_color_list):
                    img_pts = project_to_camera_lane(trans_pts, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy())
                    for j in range(img_pts.shape[0]-1):
                        x1 = int(img_pts[j][0])
                        y1 = int(img_pts[j][1])
                        x2 = int(img_pts[j+1][0])
                        y2 = int(img_pts[j+1][1])
                        try:
                            cv2.line(image, (x1, y1), (x2, y2), color, 3)
                        except:
                            pass
                image = draw_dict_top_right(image, aug_info)
                img_list.append(image)
                img_tmp = cv2.hconcat(img_list)
            image_full = cv2.vconcat([image_full, img_tmp])
            img_list = []
            for j in range(4,7):
                image = imgs[idx][0][j].cpu().numpy()
                aug_info = img_meta['pos_aug_info'][j] if 'pos_aug_info' in img_meta else {}
                intrinsic = intrinsics[idx][0][j]
                extrinsic = extrinsics[idx][0][j]
                mean = np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
                std = np.array([58.395, 57.12, 57.375]).reshape(3, 1, 1)
                image = ((image * std) + mean)[[2, 1, 0], :, :].transpose(1, 2, 0).astype(np.uint8).copy()
                for trans_pts,color in zip(all_pts_list,all_color_list):
                    img_pts = project_to_camera_lane(trans_pts, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy())
                    for j in range(img_pts.shape[0]-1):
                        x1 = int(img_pts[j][0])
                        y1 = int(img_pts[j][1])
                        x2 = int(img_pts[j+1][0])
                        y2 = int(img_pts[j+1][1])
                        try:
                            cv2.line(image, (x1, y1), (x2, y2), color, 3)
                        except:
                            pass
                image = draw_dict_top_right(image, aug_info)
                img_list.append(image)
                img_tmp = cv2.hconcat(img_list)
            image_full = cv2.vconcat([image_full, img_tmp])
            # # for raw image
            # img_list = []
            # for j in [0,1,2]:#range(1,4):
            #     image = imgs[j].cpu().numpy()
            #     # intrinsic = intrinsics[idx][0][j]
            #     intrinsic = intrinsics[j].cpu().numpy() if isinstance(x, torch.Tensor) else intrinsics[j]
            #     extrinsic = extrinsics[j]
            #     for trans_pts,color in zip(all_pts_list,all_color_list):
            #         img_pts = project_to_camera_lane(trans_pts, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy())
            #         for j in range(img_pts.shape[0]-1):
            #             x1 = int(img_pts[j][0])
            #             y1 = int(img_pts[j][1])
            #             x2 = int(img_pts[j+1][0])
            #             y2 = int(img_pts[j+1][1])
            #             try:
            #                 cv2.line(image, (x1, y1), (x2, y2), color, 3)
            #             except:
            #                 pass
            #     img_list.append(image)
            #     img_tmp = cv2.hconcat(img_list)
            # image_full = cv2.vconcat([image_full, img_tmp.astype(image_full.dtype)])
            # img_list = []
            # for j in range(4,7):
            #     image = imgs[j].cpu().numpy()
            #     # intrinsic = intrinsics[idx][0][j]
            #     # intrinsic = img_meta['intrinsics'][j]
            #     intrinsic = intrinsics[j].cpu().numpy() if isinstance(x, torch.Tensor) else intrinsics[j]
            #     extrinsic = extrinsics[j]
            #     for trans_pts,color in zip(all_pts_list,all_color_list):
            #         img_pts = project_to_camera_lane(trans_pts, intrinsic[:3,:3].cpu().numpy(), extrinsic.cpu().numpy())
            #         for j in range(img_pts.shape[0]-1):
            #             x1 = int(img_pts[j][0])
            #             y1 = int(img_pts[j][1])
            #             x2 = int(img_pts[j+1][0])
            #             y2 = int(img_pts[j+1][1])
            #             try:
            #                 cv2.line(image, (x1, y1), (x2, y2), color, 3)
            #             except:
            #                 pass
            #     img_list.append(image)
            #     img_tmp = cv2.hconcat(img_list)
            # image_full = cv2.vconcat([image_full, img_tmp.astype(image_full.dtype)])
            dst_dir = "./work_dirs/visualize_gt/2025_05_15_onlycrop/"
            os.makedirs(dst_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dst_dir, image_path.replace('/','-')), image_full)
        return

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """

        # self._data = copy.deepcopy(data)
        if return_loss:
            pp_heavy_keys = ['voxels', 'voxel_coords', 'voxel_num_points', 'voxel_label', 'anchors', 'box_labels', 'reg_targets', 'reg_weights']
            collect_keys = ['gt_bboxes_3d', 'gt_labels_3d', 'gt_motions_3d',
                            'gt_bboxes_2d', 'gt_labels_2d', 'gt_2d_to_3d_idx', 'gt_cam_idx',
                            'gt_bboxes_3d_cam', 'gt_centers_2d', 'gt_depths_cam', 
                            'img_metas', 'gt_plan_pos', 'gt_obj_trajs',
                            'map_gt_bboxes_3d', 'map_gt_labels_3d',
                            'gt_seg_mask', 'gt_seg_offset', 'gt_seg_type', 'gt_seg_color', 'gt_pv_seg_mask',
                            'occ_voxel_semantics', 'occ_voxel_instances','depth_map',
                            'occ_instance_class_ids', 'occ_camera_masks',
                            'pred_bbox3d_range_valid', 'points']
            for key in collect_keys:
                if key in data:
                    data[key] = list(zip(*data[key]))
            
            for key in data:
                for sub_key in pp_heavy_keys:
                    if sub_key in key and key not in collect_keys:
                        data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)


    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_motions_3d=None,
                      gt_bboxes_2d=None,
                      gt_labels_2d=None,
                      gt_2d_to_3d_idx=None,
                      gt_cam_idx=None,
                      gt_only_2d_flag=None,
                      gt_bboxes_3d_cam=None,
                      gt_centers_2d=None,
                      gt_depths_cam=None,
                      gt_bboxes_ignore=None,
                      gt_plan_pos=None,
                      gt_obj_trajs=None,
                      map_gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      gt_seg_mask=None,
                      gt_seg_offset=None,
                      gt_seg_type=None,
                      gt_seg_color=None,
                      gt_pv_seg_mask=None,
                      occ_voxel_semantics=None,
                      occ_voxel_instances=None,
                      occ_instance_class_ids=None,
                      occ_camera_masks=None,
                      data_tag=None,
                      pred_bbox3d_range_valid=None,
                      depth_map=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels_2d (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes_2d (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # len(img_metas): 1; len(img_metas[0]): 2
        # img_metas[0][0].keys(): dict_keys(['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'])
         ######  T = 1
        #随机性固定
        TARGET_SAMPLE_ID = "s3://mapless-datasets-v2-laneline-t68/to_dongfeng/road-geometry/img/sdc3-adas-3/data_collection/gt_data/pilotGtRawParser/dongfengpro_vd_gt_collection/epai7-960/2025_12/2025_12_02/2025_12_02_18_54_56_AutoCollect_pilotGtRawParser/camera/center_camera_fov120 s3://mapless-datasets-v2-laneline-t68/to_dongfeng/road-geometry/img/sdc3-adas-3/data_collection/gt_data/pilotGtRawParser/dongfengpro_vd_gt_collection/epai7-960/2025_12/2025_12_02/2025_12_02_18_54_56_AutoCollect_pilotGtRawParser/camera/center_camera_fov30/1764671070304586452.jpg"
        meta0 = img_metas
        while isinstance(meta0, (list, tuple)):
            meta0 = meta0[0]

        sample_id = f'{meta0["scene_token"]} {meta0["filename"][0]}'
        self.hit_target = sample_id == TARGET_SAMPLE_ID

        if self.hit_target and self.debug_print:
            print(...)
        T = data['img'].size(1)
        # data['img'].shape: torch.Size([2, 1, 7, 3, 544, 960])
        # self.draw_raw_image(data['img'][0,0,0], './img_fov30.jpg')
        # self.draw_raw_image(data['img'][0,0,1], './img_fov120.jpg')
        # self.draw_raw_image(data['img'][0,0,2], './img_left_front.jpg')
        # self.draw_raw_image(data['img'][0,0,3], './img_left_rear.jpg')
        # self.draw_raw_image(data['img'][0,0,4], './img_rear.jpg')
        # self.draw_raw_image(data['img'][0,0,5], './img_right_rear.jpg')
        # self.draw_raw_image(data['img'][0,0,6], './img_right_front.jpg')

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]  # tensor([], device='cuda:0', size=(2, 0, 7, 3, 544, 960))
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        if self.fix_backbone:
            self.eval()
            with torch.no_grad():
                rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)
                #随机性固定
                if self.hit_target and self.debug_print:
                    print("IMG_FEATS", [tensor_hash(f) for f in rec_img_feats])
                #随机性固定
            self.train()
        else:
            rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads) # the later param is just for img_feats reshape
            #随机性固定
            if self.hit_target and self.debug_print:
                print("IMG_FEATS", [tensor_hash(f) for f in rec_img_feats])
            #随机性固定
        if T - self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img.clone(), T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = []
            for i in range(len(rec_img_feats)): # i is idx of batch sample
                data['img_feats'].append(torch.cat([prev_img_feats[i], rec_img_feats[i]], dim=1)) # 同一个样本沿时间维度进行concat
        else:     ######## T=1, 逐帧训练  num_frame_backbone_grads = 1
            if self.fix_backbone:
                self.eval()
                with torch.no_grad():
                    data['img_feats'] = rec_img_feats
                self.train()
            else:  #######平时训练都是false，需要和别的模型结合才需要固定
                # actually enter here.
                data['img_feats'] = rec_img_feats
                # print(rec_img_feats[0].shape, 'rec_img_feats[0].shape')
        if 'points' in data:
            data['pts_feats'] = data['points']

        # TMP vis lane3d gt
        # self.intermediate_visual_gt(data['img'], img_metas, data['intrinsics'], data['extrinsics'], map_gt_bboxes_3d, map_gt_labels_3d, gt_bboxes_3d, occ_voxel_semantics)
        losses = self.obtain_history_memory(img_metas,
                                            gt_bboxes_3d,
                                            gt_labels_3d,
                                            gt_motions_3d,
                                            gt_bboxes_2d,
                                            gt_labels_2d,
                                            gt_2d_to_3d_idx,
                                            gt_cam_idx,
                                            gt_only_2d_flag,
                                            gt_bboxes_3d_cam,
                                            gt_centers_2d,
                                            gt_depths_cam,
                                            gt_bboxes_ignore,
                                            gt_plan_pos,
                                            gt_obj_trajs,
                                            map_gt_labels_3d,
                                            map_gt_bboxes_3d,
                                            gt_seg_mask,
                                            gt_seg_offset,
                                            gt_seg_type,
                                            gt_seg_color,
                                            gt_pv_seg_mask,
                                            occ_voxel_semantics,
                                            occ_voxel_instances,
                                            occ_instance_class_ids,
                                            occ_camera_masks,
                                            data_tag,
                                            pred_bbox3d_range_valid,
                                            depth_map,
                                            **data)

        return losses
  
    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                if isinstance(data[key][0], DC):
                    data[key] = data[key][0].data[0][0].unsqueeze(0).cuda()
                else:
                    data[key] = data[key][0][0].unsqueeze(0)
            else:
                if isinstance(data[key][0], DC):
                    if isinstance(data[key][0].data[0], list):
                        data[key] = torch.stack(data[key][0].data[0], dim=0).cuda()
                    else:   
                        data[key] = data[key][0].data[0].cuda()
                else:
                    data[key] = data[key][0]
        if isinstance(img_metas[0], DC):
            img_metas[0] = img_metas[0].data[0]
        return self.simple_test(img_metas[0], **data)
    
    def bbox3d2result(self, bboxes, scores, labels, attrs=None, motions=None, trajs=None, instance_inds=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        if motions is not None:
            result_dict['motions_3d'] = motions.cpu()

        if trajs is not None:
            result_dict['obj_trajs'] = trajs.cpu()

        if instance_inds is not None:
            result_dict['instance_inds'] = instance_inds.cpu()

        return result_dict

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        data['position_level'] = self.occ_levels    # 0
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, img_metas, **data)

        topk_indexes = outs_roi['topk_indexes']
        outs_rpn = self.forward_rpn_head(img_metas, **data)
        rpn_proposals = None
        nv2d_query_ref_points_pvb = None
        if self.with_img_rpn and outs_rpn is not None:
            outs_rpn += (None,)
            rpns_pinhole_pvb, scores_pinhole_pvb, info_pinhole_pvb, class_scores_pinhole_pvb, _ = self.img_rpn_head.get_reference_points(
                    *outs_rpn,
                    img_metas,
                    camera_type='pinhole',
                    cfg=None,
                    quant_restrict_ref_pts=True,
                    rescale=False,
                    all_cls_scores_2d=None,
                    **data
                )
            # mono_coord, mono_score, mono_cls = self.img_rpn_head.decode_mono3d(rpns_pinhole_pvb, info_pinhole_pvb, 
            #                                                                    scores_pinhole_pvb, class_scores_pinhole_pvb, 
            #                                                                    cam_idx=1, threshold_2d=0.1, distanct_mask=-10000)
            ref_pts = []
            cam_nums = data['img'].shape[1]
            pc_range_ = self.pts_bbox_head.bbox_coder.pc_range
            for kk in range(len(img_metas)):
                ret_pts_metakk = torch.cat(rpns_pinhole_pvb[kk*cam_nums:kk*cam_nums+cam_nums], dim=0)[:,:3]
                ret_pts_metakk[:,0] = (ret_pts_metakk[:,0] - pc_range_[0]) / (pc_range_[3] - pc_range_[0])
                ret_pts_metakk[:,1] = (ret_pts_metakk[:,1] - pc_range_[1]) / (pc_range_[4] - pc_range_[1])
                ret_pts_metakk[:,2] = (ret_pts_metakk[:,2] - pc_range_[2]) / (pc_range_[5] - pc_range_[2])
                ref_pts.append(ret_pts_metakk.unsqueeze(0))
            nv2d_query_ref_points_pvb = torch.cat(ref_pts, dim=0)

        # lidar query generation
        if self.with_pts_backbone:
            if 'SingleStageFSDV2' in str(type(self.pts_backbone)) or 'FastPillar' in str(type(self.pts_backbone)):
                out_dict = self.pts_backbone.simple_test(data['pts_feats'], img_metas)
            else:
                batch_dict = {}
                for key in ['points', 'voxels', 'voxel_coords', 'voxel_num_points']:
                    val = data[key]
                    if key in ['voxels', 'voxel_num_points']:
                        batch_dict[key] = val[0]
                    elif key in ['points', 'voxel_coords']:
                        coors = []
                        for i, coor in enumerate(val):   # coor: (Ni, C)
                            idx_col = torch.full((coor.size(0), 1), i, dtype=coor.dtype, device=coor.device)
                            coor_pad = torch.cat([idx_col, coor], dim=1)  # (Ni, C+1)
                            coors.append(coor_pad)
                        batch_dict[key] = torch.cat(coors, dim=0)
                for key in data:
                    for sub_key in ['anchors', 'box_labels', 'reg_targets', 'reg_weights']:
                        if sub_key in key:
                            batch_dict[key] = data[key]
                out_dict = self.pts_backbone.forward_rpn(return_loss=False, **batch_dict)
                if os.getenv("SAVE_TENSOR") == 'True' and self.frame_id == 175:
                    self.save_dir = os.getenv("SAVE_TENSOR_PATH", "/mnt/afs2/chenxuepan/work_dirs/save_tensors")
                    output_keys = ['voxel_feats', 'query_feats', 'query_xyz', 'query_pred']
                    for k in output_keys:
                        v = out_dict.get(k, None)
                        if v is None:
                            continue
                        if isinstance(v, torch.Tensor):
                            save_tensor(v, self.save_dir, k, "far3d_head_align")
                        elif isinstance(v, (list, tuple)):
                            for i, t in enumerate(v):
                                if isinstance(t, torch.Tensor):
                                    save_tensor(t, self.save_dir, k, "far3d_head_align")
                self.frame_id += 1
        else:
            out_dict = None
        
        if self.pts_bbox_head is not None:
            if out_dict is not None and self.pts_query_generator:
                pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
                    out_dict['voxel_feats'], out_dict['voxel_coors'], out_dict['voxel_xyz'], out_dict['query_feats'],
                    out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], 1)     
            else:
                pts_feat, pts_pos, pts_query_feat, pts_query_center = None, None, None, None

            if "FarHead" in str(type(self.pts_bbox_head)):
                outs = self.pts_bbox_head(img_metas, outs_roi, pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                          pts_feat=pts_feat, pts_pos=pts_pos, pts_shape=None, **data)
            else:
                location_new = location.clone() if isinstance(location, torch.Tensor) else [t.clone() for t in location]
                if self.pvb_position_encoder is None:
                    outs = self.pts_bbox_head(location_new, None, None, img_metas, topk_indexes, rpn_proposals, nv2d_query_ref_points_pvb, **data)
                else:
                    pvb_pe, pvb_cone = self.pvb_position_encoder(data, location_new, topk_indexes, img_metas)
                    outs = self.pts_bbox_head(None, pvb_pe, pvb_cone, img_metas, topk_indexes, rpn_proposals, nv2d_query_ref_points_pvb, **data)
            

        ## gop
        # data.update(pillar_features=out_dict['pillar_features'])
        spatial_features_2d_lidar = out_dict['spatial_features_2d_lidar'] if out_dict is not None else None
        data.update(spatial_features_2d_lidar_gop_driving=spatial_features_2d_lidar)
        outs_gop = self.forward_gop_head(data, img_metas)

        ## bev encoder for lane and occ
        if self.bev_encoder:
            if (hasattr(self.bev_encoder, 'encoder') and isinstance(self.bev_encoder.encoder, BEVFormerEncoderStream)):
                if img_metas[0]['scene_token'] != self.prev_scene_token:
                    self.prev_scene_token = img_metas[0]['scene_token']
                    data['prev_exists'] = data['img'].new_zeros(1)
                    self.bev_encoder.encoder.reset_memory()                  
            mlvl_feats = [data['img_feats'][level] for level in self.occ_levels]
            bev_embed, bev_pos, depth = self.bev_encoder(
                mlvl_feats,
                lidar_feat = spatial_features_2d_lidar,
                intrinsics = data['intrinsics'],
                extrinsics = data['extrinsics'],
                lidar2img = data['lidar2img'],
                img_metas=img_metas,
                data=data,
                )
            #随机性固定
            if self.hit_target and self.debug_print:
                print("BEV_EMBED", tensor_hash(bev_embed))
            #随机性固定
            data['bev_embed'] = bev_embed

        ## lane3d
        if self.lane3d_head is not None:
            location_new = location.clone() if isinstance(location, torch.Tensor) else [t.clone() for t in location]
            # if self.lane3d_position_encoder is not None:
            #     lane3d_pe, lane3d_cone = self.lane3d_position_encoder(data, location, topk_indexes, img_metas)
            # else:
            #     lane3d_pe, lane3d_cone = None, None
            # data = data.copy()  ####避免data被原地修改,传入后续pts_bbox_head报错
            data['img_metas'] = img_metas
            lane3d_outs = self.lane3d_head(None, None, **data)
            data['map_queries'] = lane3d_outs['map_queries']
            if lane3d_outs is not None and 'bev_embed' in lane3d_outs:
                data['bev_embed'] = lane3d_outs['bev_embed']
            # if os.getenv("DUMP_ROAD") == 'True' and not getattr(self, "dumped_once", False):
            if os.getenv("DUMP_ROAD") == 'True' and self.dump_road_count < self.MAX_DUMP:
                self.dumped_once = True   # 确保只 dump 一次
                lane3d_outs_dump = deepcopy(lane3d_outs)
                pt_dir = 'dump_datas/laneline_3d/pt/'
                bin_dir = 'dump_datas/laneline_3d/bin/'
                os.makedirs(pt_dir, exist_ok=True)
                os.makedirs(bin_dir, exist_ok=True)

                # -------- Save img_feats[2] --------
                img_feats_list = data.get('img_feats', None)
                img_feats_tensor = img_feats_list[2]
                img_feats_tensor2 = img_feats_tensor.view(img_feats_tensor.shape[0], -1, img_feats_tensor.shape[3], img_feats_tensor.shape[4])
                print(img_feats_tensor2.shape)
                if isinstance(img_feats_list, (list, tuple)) and len(img_feats_list) > 0:
                    save_tensor_if_valid(
                        tensor=img_feats_tensor2,
                        pt_path=os.path.join(pt_dir, "img_feats_2.pt"),
                        bin_prefix=os.path.join(bin_dir, "img_feats_2"),
                        idx = self.dump_road_count
                    )
                
                if not self.lc_fusion and (self.bev_encoder.fuser or self.bev_encoder.lidar_proj):
                    save_tensor_if_valid(
                        tensor=data.get('spatial_features_2d_lidar_gop_driving', None),
                        pt_path=os.path.join(pt_dir, "spatial_features_2d_lidar.pt"),
                        bin_prefix=os.path.join(bin_dir, "spatial_features_2d_lidar"),
                        idx = self.dump_road_count
                    )

                # -------- Save reference_points_cam --------
                ref_points = data.get('reference_points_cam', None)
                save_tensor_if_valid(
                    tensor=ref_points,
                    pt_path=os.path.join(pt_dir, "reference_points_cam.pt"),
                    bin_prefix=os.path.join(bin_dir, "reference_points_cam"),
                    idx = self.dump_road_count
                )
                # -------- Save bev_embed --------
                bev_embed = lane3d_outs_dump.get('bev_embed', None)
                save_tensor_if_valid(
                    tensor=bev_embed,
                    pt_path=os.path.join(pt_dir, "bev_embed.pt"),
                    bin_prefix=os.path.join(bin_dir, "bev_embed"),
                    idx = self.dump_road_count
                )
                # -------- Get lane3d_outs 的 batch 最后一帧 --------
                lane3d_outs_last = {}
                outputs_seg = lane3d_outs_dump['seg']
                outputs_seg_type = lane3d_outs_dump['seg_type']
                outputs_seg_color = lane3d_outs_dump['seg_color']
                outputs_seg_offest = lane3d_outs_dump['seg_offset']
                line_heatmaps = outputs_seg.sigmoid() 
                seg_types = outputs_seg_type.sigmoid()
                seg_colors = outputs_seg_color.sigmoid()
                bs, channel_num, H, W = line_heatmaps.shape
                seg_offsets = outputs_seg_offest.view(bs, channel_num, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
                seg_types = seg_types.permute(0, 2, 3, 1).contiguous()
                seg_colors = seg_colors.permute(0, 2, 3, 1).contiguous()
                lane3d_outs_dump['seg'] = line_heatmaps
                lane3d_outs_dump['seg_type'] = seg_types
                lane3d_outs_dump['seg_color'] = seg_colors
                lane3d_outs_dump['seg_offset'] = seg_offsets
                for k, v in lane3d_outs_dump.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] > 0 and k.startswith("seg") == False:
                        lane3d_outs_last[k] = v[-1]
                        # print(f"[INFO] lane3d_outs[{k}][-1] shape: {v[-1].shape}")
                    else:
                        lane3d_outs_last[k] = v
                        # print(f"[SKIP] lane3d_outs[{k}] is not tensor or empty")
                # -------- Save all map_all_* 和 seg* fields --------
                for key, value in lane3d_outs_last.items():
                    if not isinstance(value, torch.Tensor):
                        continue

                    if key in MAP_KEY_MAPPING:  # map_all_* 特殊处理 + 重命名
                        new_key = MAP_KEY_MAPPING[key]
                        save_tensor_if_valid(
                            tensor=value,
                            pt_path=os.path.join(pt_dir, f"{new_key}.pt"),
                            bin_prefix=os.path.join(bin_dir, new_key),
                            idx = self.dump_road_count
                        )

                    elif key.startswith("seg"):  # seg* 正常保存
                        save_tensor_if_valid(
                            tensor=value,
                            pt_path=os.path.join(pt_dir, f"{key}.pt"),
                            bin_prefix=os.path.join(bin_dir, key),
                            idx = self.dump_road_count
                        )
                self.dump_road_count += 1

                        
        ## occ3d
        if self.occ3d_head is not None:
            occ3d_outs = self.forward_occ3d_head(img_metas, data)
            data['bev_embed'] = lane3d_outs.get('bev_embed', None)
            if os.getenv("DUMP_OCC") == 'True' and self.dump_occ_count < self.MAX_DUMP:
                pt_dir = 'dump_datas/bev_occ/pt/'
                bin_dir = 'dump_datas/bev_occ/bin/'
                os.makedirs(pt_dir, exist_ok=True)
                os.makedirs(bin_dir, exist_ok=True)
                # -------- Save bev_embed --------
                save_tensor_if_valid(
                    tensor=data.get('bev_embed', None),
                    pt_path=os.path.join(pt_dir, "bev_embed.pt"),
                    bin_prefix=os.path.join(bin_dir, "bev_embed"),
                    idx = self.dump_occ_count
                )
                # -------- Save lidar feat ------------
                save_tensor_if_valid(
                    tensor=data.get('spatial_features_2d_lidar_gop_driving', None),
                    pt_path=os.path.join(pt_dir, "spatial_features_2d_lidar.pt"),
                    bin_prefix=os.path.join(bin_dir, "spatial_features_2d_lidar"),
                    idx = self.dump_occ_count
                )
                # -------- Save all outputs from occ3d_outs --------
                for key, value in occ3d_outs.items():
                    save_tensor_if_valid(
                        tensor=value,
                        pt_path=os.path.join(pt_dir, f"{key}.pt"),
                        bin_prefix=os.path.join(bin_dir, key),
                        idx = self.dump_occ_count
                    )
                self.dump_occ_count += 1

        ## planner
        if self.planning_head is not None:
            plan_pos = self.planning_head(outs['memory'], outs['pos_embed'], outs['object_queries'], **data) # B,100,N,1
            plan_pos = plan_pos['plan_coords_preds']
            B, _ , N, _ = plan_pos.shape
            plan_pos = plan_pos.reshape(B, -1, 2, N)
            if self.pred_delta:
                for i in range(plan_pos.shape[1] - 1):
                    plan_pos[:, i+1] = plan_pos[:, i] + plan_pos[:, i+1]
            else:
                for i in range(plan_pos.shape[1]):
                    plan_pos[:, i] *= (i + 1)
            if self.pred_frame_tf > 0:
                obj_trajs_tf = plan_pos[..., 1:].clone()
                obj_trajs_tf = obj_trajs_tf.permute(0, 3, 1, 2).contiguous()
            plan_pos = plan_pos[..., 0:1]
        
        bbox_results = []
        if self.pts_bbox_head is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
            for i, (bboxes, scores, labels, mask, motions, trajs, instance_inds) in enumerate(bbox_list):
                bbox_result = self.bbox3d2result(bboxes, scores, labels, None, motions, trajs, instance_inds)
                if self.pred_frame_tf > 0:
                    obj_trajs_tf_mask = obj_trajs_tf[i][mask]
                    bbox_result['obj_trajs_tf'] = obj_trajs_tf_mask.cpu()
                    for j in range(len(bbox_result['obj_trajs_tf'])):
                        obj_pos = bbox_result['boxes_3d'].tensor[j].numpy()[:2]
                        obj_yaw = bbox_result['boxes_3d'].tensor[j].numpy()[6]
                        rot = np.array([[math.cos(obj_yaw), math.sin(obj_yaw)],
                                        [-math.sin(obj_yaw), math.cos(obj_yaw)]])
                        bbox_result['obj_trajs_tf'][j] = bbox_result['obj_trajs_tf'][j] @ rot + obj_pos
                bbox_results.append(bbox_result)
        if self.multimodal_gop is not None:
            if os.getenv('draw_dense_gop', False):
                for gop_task in self.multimodal_gop.multimodal_head.model_cfg.tasks:
                    gop_reults = self.multimodal_gop.multimodal_head.get_results(gop_task, 'gop_driving')
                    for i in range(len(gop_reults)):
                        bbox_results[i][f"{gop_task}_gop_driving_boxes_3d"] = gop_reults[i]['boxes_3d']
                        bbox_results[i][f"{gop_task}_gop_driving_labels_3d"] = gop_reults[i]['labels_3d']
                        bbox_results[i][f"{gop_task}_gop_driving_scores_3d"] = gop_reults[i]['scores_3d']
            if self.pts_bbox_head_gop is not None:
                bbox_list = self.pts_bbox_head_gop.get_bboxes(outs_gop, img_metas)
                for i, (bboxes, scores, labels, mask, motions, trajs, instance_inds) in enumerate(bbox_list):
                    bbox_result = self.bbox3d2result(bboxes, scores, labels, None, motions, trajs, instance_inds)
                    bbox_results[i].update(dict(
                        sparse_gop_driving_boxes_3d = bbox_result['boxes_3d'],
                        sparse_gop_driving_labels_3d = bbox_result['labels_3d'],
                        sparse_gop_driving_scores_3d = bbox_result['scores_3d'],
                    ))
        if self.lane3d_head is not None:
            map_bbox_list = self.lane3d_head.get_bboxes(lane3d_outs, img_metas)
            if self.seg_res:
                seg_list = self.lane3d_head.get_seg_res(lane3d_outs)
            # assert len(map_bbox_list) == B
            for i in range(len(map_bbox_list)):
                # map_bboxes, map_scores, map_labels, map_pts, map_pts_types, map_pts_colors = map_bbox_list[i]
                map_scores, map_labels, map_pts, map_pts_types, map_pts_colors = map_bbox_list[i][:5]
                map_pts_uncertain = None
                if len(map_bbox_list[i]) > 5:
                    map_pts_uncertain = map_bbox_list[i][5]
                seg = None
                pv_seg = None
                if self.seg_res:
                    seg, pv_seg, seg_offset, seg_type, seg_color = seg_list[i]
                # map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts, map_pts_types, map_pts_colors,
                map_bbox_result = self.map_pred2result(map_scores, map_labels, map_pts, map_pts_types, map_pts_colors,  # clw modify
                                                        seg=seg,pv_seg=pv_seg, pts_uncertain=map_pts_uncertain)
                if self.seg_res:
                    flop_count_mode = data['flop_count_mode'] if 'flop_count_mode' in data else False
                    seg_ins_result = self.calc_instance_from_seg(seg, seg_offset, seg_type, seg_color, flop_count_mode=flop_count_mode)
                    map_bbox_result.update(seg_ins_result)
                if self.pts_bbox_head is not None:
                    bbox_results[i].update(map_bbox_result)
                else:
                    bbox_results.append(map_bbox_result)

        if self.occ3d_head is not None:
            occ3d_res = self.occ3d_head.merge_occ_pred(occ3d_outs)
            for i in range(len(occ3d_res)):
                bbox_results[i].update(occ3d_res[i])

        if rpn_proposals is not None:
            assert len(bbox_results) == len(rpn_proposals[0]) == len(rpn_proposals[1]) == len(rpn_proposals[2])
            for i in range(len(bbox_results)):
                bbox_results[i].update(reference_points=rpn_proposals[0][i].cpu())
                bbox_results[i].update(top_proposals_bboxes_2d=rpn_proposals[1][i].cpu())
                bbox_results[i].update(top_proposals_scores=rpn_proposals[2][i].cpu())
        
        if self.planning_head is not None:
            assert len(bbox_results) == B
            for i in range(len(bbox_results)):
                bbox_results[i].update(plan_pos=plan_pos[i].cpu())
        return bbox_results
    
    def calc_instance_from_seg(self, seg, seg_offset, seg_type, seg_color, flop_count_mode):
        result_dict = dict()
        if flop_count_mode:
            result_dict['seg_offset'] = seg_offset
            result_dict['seg_type'] = seg_type
            result_dict['seg_color'] = seg_color
            return result_dict
        if seg is None:
            return result_dict
        map_seg_processor = MapSegPostProcessor(map_range=self.map_range)
        line_heatmaps = seg.sigmoid().cpu()
        channel_num, H, W = line_heatmaps.shape
        seg_offsets = seg_offset.view(channel_num, 2, H, W).permute(0, 2, 3, 1).contiguous().cpu()
        seg_types = seg_type.sigmoid().permute(1, 2, 0).contiguous().cpu()
        seg_colors = seg_color.sigmoid().permute(1, 2, 0).contiguous().cpu()
        line_list = []
        label_list = []
        type_list = []
        color_list =[]
        for i in range(channel_num):
            if i == 2:
                continue
            res = map_seg_processor.heatmap_to_polyline(
                line_heatmaps[i], seg_offsets[i], seg_types, seg_colors, i)
            line_xy_list, line_type_list, line_color_list = res
            line_list.extend(line_xy_list)
            label_list.extend([i] * len(line_xy_list))
            type_list.extend(line_type_list)
            color_list.extend(line_color_list)
        result_dict['seg_line_pts'] = line_list
        result_dict['seg_line_labels'] = label_list
        result_dict['seg_line_types'] = type_list
        result_dict['seg_line_colors'] = color_list
        # result_dict['seg_offsets'] = seg_offsets
        return result_dict
    
    def map_pred2result(self, scores, labels, pts, pts_types, pts_colors, attrs=None, 
                        seg=None, pv_seg=None, pts_uncertain=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            # map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'),
            map_pts_types_3d=pts_types.to('cpu'),
            map_pts_colors_3d=pts_colors.to('cpu'))
        if pts_uncertain is not None:
            result_dict['map_pts_3d_uncertain'] = pts_uncertain.to('cpu')
        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()
        if seg is not None:
            result_dict['seg'] = seg.sigmoid().cpu()
        if pv_seg is not None:
            result_dict['pv_seg'] = pv_seg.sigmoid().cpu()

        return result_dict

    def draw_raw_image(self, 
                       img_array, 
                       img_name = './image_raw.jpg',
                       mean = [123.675, 116.28, 103.53], 
                       std = [58.395, 57.12, 57.375]):
        img_array = np.transpose(img_array.cpu().numpy(), (1, 2, 0))   # 调整通道顺序为 (height, width, channel)
        img_array = img_array * std + mean
        cv2.imwrite(img_name, img_array)  
    
    def replace_img(self, img_name, mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], **data):
        import mmcv
        img = cv2.imread(img_name)  # (2160, 3840, 3)
        img = mmcv.imresize(img, (960, 540), return_scale=False)
        img = (img - mean) / std
        padded_img = mmcv.impad_to_multiple(img, 32, pad_val=0)
        data['img'][0,1] = torch.tensor(padded_img).cuda().permute(2,0,1).contiguous()
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        # data['img'].shape: torch.Size([1, 7, 3, 544, 960])
        # self.replace_img('1.jpg', **data)
        # self.draw_raw_image(data['img'][0,1])
        
        # data['img_feats'] = self.extract_img_feat(data['img'], 1)
        if os.getenv("COMPARE") == 'True':
            self.timestamp = img_metas[0]['filename'][0].split('/')[-1].split('.')[0]

        rec_img_feats= self.extract_img_feat(data['img'], 1)
        data['img_feats'] = rec_img_feats

        if 'points' in data:
            rec_points = [data['points'].squeeze(0)]
            data['pts_feats'] = rec_points
        if os.getenv("SAVE_TENSOR") == 'True' and self.frame_id == 175:
            self.save_dir = os.getenv("SAVE_TENSOR_PATH", "/mnt/afs2/chenxuepan/work_dirs/save_tensors")
            save_tensor(data['img_feats'][0], self.save_dir, "img_feats", "img_backbone")
            save_tensor(data['pts_feats'][0], self.save_dir, "points", "preprocess_points")
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        self.frame_id += 1
        return bbox_list

    def forward_dummy_backbone(self,image_data):
        img_feats = self.img_backbone(image_data)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        if isinstance(self.position_level, list):
            outs = {}
            for idx in self.position_level:
                outs[f"img_feats_{idx}"] = img_feats[idx]
        else:
            outs = {
                'img_feats': img_feats[self.position_level],
            }

        if self.with_img_rpn:
            import torch.nn.functional as F
            H=544
            W=960
            img_meta = dict(
                img_shape=[[H, W, 3] for _ in range(7)],
                pad_shape=[[H, W, 3]]
            )
            img_metas = [img_meta, ]
            for img_meta in img_metas:
                img_meta.update(input_shape=(H,W))
            data={}
            data['img_feats'] = img_feats
            outs_rpn = self.img_rpn_head(img_metas, **data)
            cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, bbox2d_pred, scales = outs_rpn
            assert len(cls_score) == len(bbox_pred) == len(dir_cls_pred) == len(
                centerness) == len(bbox2d_pred), "fpn levels length must be same"   

            mlvl_levels = len(cls_score)
            outputs, outputs_scales = {}, {}
            outputs_scales['scale_offset'] = []
            outputs_scales['scale_depth'] = []
            outputs_scales['scale_size'] = []
            outputs_scales['scale_bbox2d'] = []

            all_lvl_scores = []
            for i in range(mlvl_levels):
                # bbox scale
                scale_offset, scale_depth, scale_size, scale_bbox = scales[i]
                # cls score
                lvl_cls_score = cls_score[i].sigmoid()
                # centerness
                lvl_centerness = centerness[i].sigmoid()
                # outputs['fpn{}.centerness'.format(i)] = lvl_centerness

                # nms
                nms_kernel_size = 3
                lvl_cls_score = lvl_cls_score * lvl_centerness
                outputs['fpn{}.cls_score'.format(i)] = lvl_cls_score
                
                pad = nms_kernel_size // 2
                objectness_label = torch.max(lvl_cls_score, dim=1, keepdim=True)[0]
                hmax = F.max_pool2d(objectness_label, kernel_size=nms_kernel_size, stride=1, padding=pad)
                keep = torch.gt(hmax,objectness_label).to(dtype=torch.float32)
                keep = torch.ones_like(keep).to(keep.device) - keep
                objectness_label = objectness_label * keep
                all_lvl_scores.append(objectness_label.flatten(-3))

                # bbox 3d
                clvl_bbox_pred = bbox_pred[i]
                # outputs['fpn{}.bbox_pred'.format(i)] = clvl_bbox_pred
                outputs['fpn{}.center'.format(i)] = clvl_bbox_pred[0]
                outputs['fpn{}.depth'.format(i)] = clvl_bbox_pred[1]
                # outputs['fpn{}.size'.format(i)] = clvl_bbox_pred[2]
                # outputs['fpn{}.dir'.format(i)] = clvl_bbox_pred[3]
                # # bbox 2d
                # outputs['fpn{}.bbox2d_pred'.format(i)] = bbox2d_pred[i]
                # # yaw
                # outputs['fpn{}.dir_cls_pred'.format(i)] = dir_cls_pred[i]

                # if attr_pred:
                #     outputs['fpn{}.attr_pred'.format(i)] = attr_pred[i]

                outputs_scales['scale_offset'].append(float(scale_offset.scale))
                outputs_scales['scale_depth'].append(float(scale_depth.scale))
                outputs_scales['scale_size'].append(float(scale_size.scale))
                outputs_scales['scale_bbox2d'].append(float(scale_bbox.scale))
            
            # topk
            topk_num = 20
            all_lvl_scores = torch.cat(all_lvl_scores, dim=-1)
            _, topk_indexes = torch.topk(all_lvl_scores, topk_num, dim=-1)
            outputs['topk_indexes'] = topk_indexes.to(dtype=torch.float32).reshape(topk_indexes.shape[0],1,1,-1)

            import mmcv
            mmcv.dump(outputs_scales, "scales.json")

            outs.update(outputs)
        return outs

    def forward_dummy_head(self,tmp_data,img_metas):
        if "FarHead" in str(type(self.pts_bbox_head)):
            outs = self.pts_bbox_head(img_metas, None, **tmp_data)
        else:
            topk_indexes = None
            cone = tmp_data.pop('cone')
            pos_embed = tmp_data.pop('pos_embed')
            location = self.prepare_location(img_metas, **tmp_data)
            outs = self.pts_bbox_head(location, pos_embed, cone, img_metas, topk_indexes, **tmp_data)

        return outs

    def forward_dummy_lc_head(self,tmp_data,img_metas,out_dict):
        query_cat = []
        voxel_feats = torch.randn(1, 120,128,96).cuda()
        pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
                    voxel_feats, None, None, out_dict['query_feats'],
                    out_dict['query_xyz'], out_dict['query_pred'], query_cat, 1)
        outs = self.pts_bbox_head(img_metas, None, pts_query_center,pts_query_feat,pts_feat,pts_pos,**tmp_data)
        return outs

    def forward_dummy_pts_backbone(self,data):
        batch_dict = {}
        for key in ['vfe_input', 'voxel_coords', 'voxel_valid_flag']:
            batch_dict[key] = data[key]

        # out_dict = self.pts_backbone.forward_rpn(return_loss=False, **batch_dict)
        # return {
        #     'rpn_cls_preds': out_dict['rpn_cls_preds'],
        #     'rpn_box_preds': out_dict['rpn_box_preds'],
        #     'rpn_dir_cls_preds': out_dict['rpn_dir_cls_preds'],
        #     'spatial_features_2d_pvb': out_dict['spatial_features_2d_pvb'],
        # }
        module_list = ['PillarVFE', 'PointPillarScatter_Seg', 'BaseBEVBackbone_FPN','pp_heavy_head']
        
        # onnx_outputs = [
        #     ['spatial_features_2d_lidar'],
        #     ['spatial_features_2d_cam'],
        #     ['det_pred_dicts_lidar'],
        #     ['det_pred_dicts_cam'],
        #     ['det_pred_dicts_fusion'],
        # ]
        out_dict = self.pts_backbone.forward_onnx(batch_dict, module_list, None,'pp_heavy_head')
        return {
            'final_box_dicts': out_dict['final_box_dicts'],
            'rpn_cls_preds': out_dict['det_pred_dicts_pvb']['rpn_cls_preds'],
            'rpn_box_preds': out_dict['det_pred_dicts_pvb']['rpn_box_preds'],
            'rpn_dir_cls_preds': out_dict['det_pred_dicts_pvb']['rpn_dir_cls_preds'],
            'spatial_features_2d_pvb': out_dict['spatial_features_2d_pvb'].permute(0, 2, 3, 1),
            'spatial_features_2d_lidar': out_dict['spatial_features_2d_lidar_pvb'],
        }

    def forward_dummy_multimodal_gop(self, batch_dict, onnx_module=None, onnx_outputs=None, task=None):
        
        out_dict = self.multimodal_gop.forward_onnx(batch_dict, onnx_module, onnx_outputs, task)
        
        return out_dict

    def forward_dummy_planning_head(self, memory, pos_embed, object_queries, **tmp_data):

        outs = self.planning_head(memory, pos_embed, object_queries, **tmp_data) # B,100,N,1

        return outs
    
    def forward_dummy_lane2d_head(self, features):
        outs = self.lane2d_head.forward_dummy(features)
        return outs
    
    def forward_dummy_lane3d_head(self, pos_embed, cone, **tmp_data):
        if self.bev_encoder:
            mlvl_feats = tmp_data['img_feats']
            lidar_feat = tmp_data.get('spatial_features_2d_lidar', None)
            bev_embed, bev_pos, depth = self.bev_encoder(mlvl_feats, lidar_feat, **tmp_data)
            tmp_data['bev_embed'] = bev_embed
        outs = self.lane3d_head(pos_embed, cone, **tmp_data)
        return outs

    def forward_dummy(self,*data):
        image_data = []
        _,_,H,W = data[0].shape

        for idx in range(7):
            image_data.append(data[idx])
        image_data = torch.cat(image_data, dim=0) #(7,3,544,960)

        coords3d = data[7]
        memory_embedding = data[8]
        # memory_reference_point = data[9]
        emb3d_temp_reference_point = data[9]
        emb1d_memory_timestamp = data[10]
        memory_ego_motion = data[11]
        cone = data[12]

        img_feats = self.img_backbone(image_data)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        out_feats = img_feats[self.position_level].clone()
    
        img_feats_reshaped = img_feats[self.position_level].view(1, BN, C, H, W)

        tmp_data = {}
        tmp_data['img_feats'] = img_feats_reshaped
        tmp_data['coords3d'] = coords3d.squeeze(0)
        tmp_data['cone'] = cone.squeeze(0)
        tmp_data['memory_embedding'] = memory_embedding.squeeze(0)
        # tmp_data['memory_reference_point'] = memory_reference_point.squeeze(0)
        tmp_data['emb3d_temp_reference_point'] = emb3d_temp_reference_point.squeeze(0)
        tmp_data['emb1d_memory_timestamp'] = emb1d_memory_timestamp.squeeze(0)
        tmp_data['memory_ego_motion'] = memory_ego_motion.squeeze(0)
        

        img_meta = dict(
            img_shape=[[H, W, 3] for _ in range(7)],
            pad_shape=[[H, W, 3]]
        )
        img_metas = [img_meta, ]
        for img_meta in img_metas:
            img_meta.update(input_shape=(H,W))

        location = self.prepare_location(img_metas, **tmp_data)
        topk_indexes = None

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **tmp_data)

        return outs
    
    def collate_lane2d_gt(self, outs_lane2d_head, gt_lanelines):
        # 注意在训练过程中，仅仅只在featmap层面进行车道线的监督，适配featmap的直接预测输出的同时，另一方面也极大地降低计算消耗
        outs_lane2d_head['2d'] = {}
        bs = len(gt_lanelines)
        lane_gt_hms = [gt_lanelines[i][0]['center_camera_fov120']['lane_gt_hm'].clone().detach() for i in range(bs)]
        outs_lane2d_head['2d']['lane_gt_hm'] = torch.stack(lane_gt_hms).cuda()
        return outs_lane2d_head

    @staticmethod
    def object2egocar(obj_trajs_list, obj_bboxes_list):
        ret_list = []
        shape_list = []
        for obj_trajs, obj_bboxes in zip(obj_trajs_list, obj_bboxes_list):
            obj_shapes = obj_bboxes.dims # x_size, y_size, z_size -> z_size, y_size, x_size
            for idx in range(obj_trajs.shape[0]):
                yaw = obj_bboxes[idx].yaw
                pos = obj_bboxes[idx].bottom_center[:,:2]
                # yaw = obj_bboxes[idx, 6]
                # pos = obj_bboxes[idx, :2]
                rot = np.array([[math.cos(yaw), math.sin(yaw)],
                                [-math.sin(yaw), math.cos(yaw)]])
                rot = torch.tensor(rot, device=obj_trajs.device, dtype=obj_trajs.dtype)
                pos = torch.tensor(pos, device=obj_trajs.device, dtype=obj_trajs.dtype)

                obj_trajs[idx] = obj_trajs[idx] @ rot
                obj_trajs[idx] = obj_trajs[idx] + pos
            ret_list.append(obj_trajs)
            shape_list.append(obj_shapes)
            # shape_list.append(obj_shapes[:,[2,1,0]]) # x_size,y_size,z_size. -> z_size, y_size, x_size
        return ret_list, shape_list
