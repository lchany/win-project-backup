import json
import os

'''
## 20251125, 11.4.0 only camera
1、路沿挖掘数据
2、dirlossweight+关闭closeptsweight
3、富旺加速

## 20251203, 2.0.1 with lidar
1、occ head 增加lidar feat 融合
2、lane 增加10.10~11.21交付数据200w
3、lane: use_close_weight=True; close_weight_value=2.5 修复上一版近处不匹配
4、基于前融合pth, merge v11.5.0版本车道线 head&occ head权重finetune

## 2025122, 2.0.2 with lidar road版本 v12.3.0
1、occ 重刷数据(去除误占据数据)
2、lane增加20251121~20251215数据vd 8w pap:25w
3、lane加入MOVABLE_ROADSIDE类型, 模型输出需要sdk配合修改
4、车道线&occ都增加lidar前融合(bev backbone前融合lidar feat)

## 20260104 2.0.3 with lidar road版本 v12.4.0
1、增加可移动路沿数据集 Y0151-Y0185 20w
2、backbone重新训练
3、occ新增数据(badcase + cone + construction_sign) cone 5.3w + construction_sign 1.3w

## 20260105 2.0.4 with lidar road版本 v12.4.0 hotfix 解决车道线大面积漏检的问题
1、增加可移动路沿的映射
2、occ增加BEV层监督loss & 边缘loss
3、occ分层监督, 不同高度层设置不同的loss weight
4、增加lane badcase 20241110-20241211 1976帧*4; 增加20251229 左转专用10w; 增加20260105左转右转48w
5、增加激光数据的随机偏移跟旋转, 旋转(-5-5), 平移(-0.5, -0.5), 点云特征随机mask以及随机丢弃点云特征
6、wh_new_data_train_rg_clip_configs tag 改为 occ, 非同源数据
'''

_base_ = ["../../../mmdetection3d/configs/_base_/default_runtime.py"]
plugin = True
plugin_dir = [
    'projects/mmdet3d_plugin/',
    'projects/fsdv2/'
]

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# custom_hooks = [
#     dict(type='UnusedParamCheckHook'),   
# ]
workflow = [("train", 1)]
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 96  # y
bev_w_ = 160  # x

seg_bev_h_ = 96 # y
seg_bev_w_ = 280 # x

ori_occ_pc_range = [-10, -25, -1.2, 100, 25, 2.4]
target_occ_pc_range = [-10, -24, -1.2, 100, 24, 2.4]

gop_driving_keys = [
    'anchors_gop_driving', 'box_labels_gop_driving', 'reg_targets_gop_driving', 'reg_weights_gop_driving'
]
# --------------- gop ---------------
gop_point_cloud_range = [0, -7.68, -4, 102.4, 7.68, 4]
gop_anchor_offsets = [0.16, -7.52, -1.78]
gop_voxel_size = [0.16, 0.16, 8]
gop_grid_size = [
    (gop_point_cloud_range[3] - gop_point_cloud_range[0]) / gop_voxel_size[0],
    (gop_point_cloud_range[4] - gop_point_cloud_range[1]) / gop_voxel_size[1],
    (gop_point_cloud_range[5] - gop_point_cloud_range[2]) / gop_voxel_size[2],
]
gop_image_size = [576, 1024]
gop_cams = [
    "center_camera_fov30",
    "center_camera_fov120",
]
point_feature_encoding = dict(
    encoding_type = 'absolute_coordinates_encoding',
    used_feature_list = ['x', 'y', 'z', 'intensity'],
    src_feature_list = ['x', 'y', 'z', 'intensity'],
)



gop_assigner_cfg = dict(
    box_coder='ResidualCoder',
    downsample_factor=2,
    type="anchor_generator_stride",
    strides=[0.32, 0.32, 0.0],
    offsets=gop_anchor_offsets,
    rotations=[0, 1.57],
    feature_map_scale_factor=1,
    region_similarity_fn='nearest_iou_similarity',
    sample_pos_fraction=-1.0,
    sample_size=512,
)



gop_filter_points_out_fov = True
gop_data_processor = dict(
    mask_points_and_label_outside_range = dict(remove_outside_label=True),
    transform_points_and_label_to_voxels = dict(
        pillar_label = True,
        voxel_size = gop_voxel_size,
        max_points_per_voxel = 32,
        max_number_of_voxels = {'train': 100000,'test': 100000}
    )
)
# --------------- gop ---------------

## lane 3d
lane3d_types = {
    'lane': [
        "SOLID_LANE", # 0
        "DASHED_LANE",
        "LEFT_DASHED_RIGHT_SOLID",
        "LEFT_SOLID_RIGHT_DASHED",
        "DOUBLE_SOLID",
        "DOUBLE_DASHED",
        "FISHBONE_SOLID",
        "FISHBONE_DASHED",
        "THICK_SOLID",
        "THICK_DASHED",
        "VARIABLE_LANE",
        "UNKNOWN",
    ],
    'roadside': [
        "CURB", # 12
        "FENCE",
        "WALL",
        "DITCH_OR_PLANE",
        "MOVABLE_ROADSIDE", #16
        "UNKNOWN",
    ],
    'crosswalk' : [
        "SOLID_LANE", # 18
        "UNKNOWN",
    ],
    'stopline' : [
        "SOLID_LANE", # 20
        "UNKNOWN",
    ]
}

lane3d_colors = {
    'lane' : [ 
        "WHITE",
        "YELLOW",
        "RED",
        "BLUE",
        "UNKNOWN",
    ],
    'roadside' : [
        "WHITE",
        "UNKNOWN",
    ],
    'crosswalk' : [
        "WHITE",
        "UNKNOWN",
    ],
    'stopline' : [
        "WHITE",
        "UNKNOWN",
    ]
}

types_nums = [len(vals) for vals in lane3d_types.values()]
colors_nums = [len(vals) for vals in lane3d_colors.values()]

map_range = [-20, -24.0, -5.0, 120.0, 24.0, 3.0]
map_classes = ["lanelines", "roadsides", "crosswalks", "stoplines"]
map_num_vec = 120
map_fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=4,
    feat_down_sample=32,
    seg_feat_dim=64,
    pv_thickness=1,
    bev_thickness=1,
    seg_bev_h=seg_bev_h_,
    seg_bev_w=seg_bev_w_
)
sdmap_ratio = 1.5
sdmap_range = [_ * sdmap_ratio for _ in map_range]
sd_method_para = dict(n_points=10)
sdmap_map_dict = dict(
    main_road=["MAINROAD", "DIRECTIONSPIT", "NORMAL"],
    secondary_road=["SECONDARY"],
    virtual_line=[
        "INNER_CROSS_ROAD",
        "UTURN",
        "Roundabout",
        "INTERSECTION",
        "NONSTANDARD_ROUNDABOUT",
    ],
    turn=["RIGHTTURN", "LeftTurn", "ReversalLeft", "AHEAD_TURN_RIGHT"],
    elevated=["ELEVATED"],
    entrance_and_exit_connect=[
        "ENTRANCE_AND_EXIT_CONNECT",
        "MAINSECONDARYENTRANCE",
        "ParkEntrance",
        "HIGHWAY_PORT",
        "HIGHWAY_CONNECTION",
        "SPECIAL_CONNECTION",
        "IC",
        "VIRTUAL_SOLID_CONNECTION",
        "INNER_VIRTUAL_CONNECT",
        "SUNKEN_ROAD_PORT",
        "JCT",
    ],
    bridge=["BRIDGE", "MovableBridge", "CROSS_LINE_OVERPASS"],
    parking=["PA", "PARKING_OCCUPY_ROAD", "PARKING_INTERNAL_ROAD"],
    service=["SA", "TOLL"],
    ramp=["Ramp", "RAMP_BOTH_PASS"],
    tunnel=["Tunnel"],
    bus=["Bus", "TruckLane", "TAXI"],
    tide=["TIDE"],
    sunken=["SUNKEN_ROAD"],
    other=[
        "NONE",
        "Enclosure",
        "Undefined",
        "PoiConnect",
        "InnerRegion",
        "FrontDoor",
        "CONSTRUCTION",
        "OWNERSHIP",
        "STEP_ROAD",
        "MOUNTAIN_ROAD",
        "CrossInner",
        "View",
        "WalkStreet",
        "OTHER",
    ],
)

gt_shift_pts_pattern = "v6_curve"
## ------------------lane end---------------------------------
## --------------------------occ------------------------------
occ_class_names = ["Cone", "Isolation-Barrel", "Pole", "Barrier", "Triangle-Warning", "Barricade", "Construction-Signs", "Gate-Rod", "others", "obj", "ground", "free",]
occ_size = [275, 120, 9]

occ_cls_id_ratio_0616_gop = [
    0.0, 0.0212, 0.0189, 0.959  # 0616总共250w occ数据，基本能代表一般性；
                                # 这里假设obj出现频率接近0，对应权重接近 1 / log10(1.02) = 116
                                # 也就是weight可能的最大值，因为obj本身比较重要，可以大一些 
                                # 地面的权重为
]
import math                                                     
class_weights_occ = [ 1 / math.log10(1.04 + i) for i in occ_cls_id_ratio_0616_gop]      # TODO：1.02附近尝试搜索；或者设置成1.002, 拉大差距，但是因为数值过大需要除以10
class_weights_occ = [ i / min(class_weights_occ) for i in class_weights_occ]  # 背景类别归一化为1.0，
# 则 1.005对应[135.33385304930857, 26.098754819117573, 28.57813840014896, 1.0]
#  1.01对应[68.09078106596532, 22.05259878376753, 23.78095364324266, 1.0]
#  1.02对应[34.46975274599611, 16.9067582206134, 17.88646871942721, 1.0]
#  1.04对应[17.66023613112996, 11.66065745169852, 12.102732640058942, 1.0]

## -------------------------occ end---------------------------

cams = [
    "center_camera_fov30",
    "center_camera_fov120",
    "left_front_camera",
    "left_rear_camera",
    "rear_camera",
    "right_rear_camera",
    "right_front_camera",
]
num_cams = len(cams)

class_names = ["VEHICLE_CAR", "VEHICLE_TRUCK", "VEHICLE_BUS", "SPECIAL_VEHICLE", "VEHICLE_TRIKE", "BIKE_BICYCLE", "PEDESTRIAN"]
# gop_class_names = ['CONE', 'ISOLATION_BARRER','POLE', 'BARRIER','TRIANGLE_WARNING','CONSTRUCTION_SIGN', 'PARKING_LOCK', 'BARRIER_GATE','CART', 'ANIMAL','COLUMN','NO_PARKING_SIGN']
gop_class_names = ['CONE', 'POLE', 'ISOLATION_BARREL', 'TRIANGLE_WARNING', 'ANIMAL', 'GATE_ROD', 'BARRIER', 'CONSTRUCTION_SIGN']

# FSDv2 setting
voxel_size = [0.16, 0.16, 8]
bbox_pred_range = [-100.0, -50.0, -5.0, 200.0, 50.0, 3.0]
extend_bbox_pred_range = [-110.0, -60.0, -10.0, 210.0, 60.0, 10.0]
point_cloud_range = [0, -20.48, -5.0, 102.4, 20.48, 3.0]
fov120_2_center_prime = [[0, 0, 1, 2], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]
pp_heavy_anchor_offsets = [point_cloud_range[0] + voxel_size[0], point_cloud_range[1] + voxel_size[1], -1.78]
pp_heavy_downsample_factor = 2
grid_size = [
    (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0],
    (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
    (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2],
]
# It has to be a multiple of [32, 64, 128]
sparse_shape = [40, 480, 512]
target_sparse_shape = [20, 240, 256]
seg_voxel_size = (0.2, 0.2, 0.2)
virtual_voxel_size=(0.4, 0.4, 0.4)
group1 = ['VEHICLE_CAR', 'VEHICLE_TRUCK']
group2 = ['VEHICLE_BUS', 'SPECIAL_VEHICLE', 'VEHICLE_TRIKE']
group3 = ['BIKE_BICYCLE']
group4 = ['PEDESTRIAN']
group_names = [group1, group2, group3, group4]
seg_score_thresh = [0.2, ] + [0.1, ] * 3
group_lens = [len(group1), len(group2), len(group3), len(group4)]
head_group1 = class_names[:5]
head_group2 = class_names[5:]
tasks = [
    dict(class_names=head_group1),
    dict(class_names=head_group2),
]

# training hyperparameter
num_gpus = 8
import torch
# if ('4090' in torch.cuda.get_device_name()) or ('L4' in torch.cuda.get_device_name()) :
#     batch_size = 5init_cfg
# else:
#     batch_size = 18
batch_size = 16
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 4
queue_length = 1
num_frame_losses = 1

#pts_ckpt = 'weights/fsdv2-converted.pth'
img_ckpt = '/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/resnet34-b627a593.pth'

pts_ckpt =None

roi_size = 7
roi_strides = [8, 16, 32, 64]
pts_in_channels = 96
transformer_use_lidar_ca = False

fix_backbone = True  # backbone will be fixed
fix_pts_backbone = True
fix_pvb_head = True  # pvb head will be fixed
fix_light = True  # light signal will be fixed, not flick
light_frame = 20  # how many light frames to be used, 10fps

model = dict(
    type='SPetr3D',
    seg_res=True,
    box_pc_range=bbox_pred_range,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    fix_backbone=fix_backbone,
    fix_pts_backbone = fix_pts_backbone,
    fix_pvb_head = fix_pvb_head,
    stride=roi_strides,
    position_level=[0,1,2,3],
    occ_levels=[2,],
    img_backbone=dict(
        type="ResNet",
        depth=34,
        num_stages=4,
        # out_indices=(3,), # 0: (64, 136, 240); 1: (128, 68, 120); 2: (256, 34, 60); 3: (512, 17, 30)
        # 0: (64, 136, 240); 1: (128, 68, 120); 2: (256, 34, 60); 3: (512, 17, 30)
        out_indices=(0, 1, 2, 3,),
        frozen_stages=-1,
        # norm_cfg=dict(type="SyncBN", requires_grad=True, eps=1e-3, momentum=0.01),
        norm_cfg=dict(type="BN", requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        with_cp=False,
        style="pytorch",
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=img_ckpt,
            # prefix='img_backbone.', 
            map_location='cpu'
        ),
    ),
    img_neck=dict(
        type='FPN',
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        # norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_cfg=dict(type="BN", requires_grad=True, eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=img_ckpt,
            # prefix='img_neck.', 
            map_location='cpu'
        ),
    ),
    # fsdv2
    pts_backbone=dict(
        type='MultiModal_PVB_GOP',
        num_class = len(class_names),
        model_cfg = dict(
            multi_task_names = ['PVB'],    # list or none
            grid_size = grid_size,
            voxel_size = voxel_size,
            virtual_voxel_size = [
                voxel_size[0] * pp_heavy_downsample_factor, 
                voxel_size[1] * pp_heavy_downsample_factor, 
                voxel_size[2]
            ],
            point_cloud_range = point_cloud_range,
            num_rawpoint_features = 4,
            num_point_features = 32,
            depth_downsample_factor = None,
            # load checkpoints info
            to_cpu = False,
            with_head = 0,
            lidar_pretrained = None,
            camera_pretrained = None,
            load_from = None,
            fuser_later = True,
            use_extra_downsample = (True and transformer_use_lidar_ca),
            freeze = dict(
                fix_modules = ['vfe', 'map_to_bev_module', 'backbone_2d_pvb'],
                ignore_keys = None,
            ),
            vfe = dict(
                type = 'PillarVFE',
                with_distance = False,
                # use_abslote_xyz = True,
                use_norm = True,
                num_point_features = 4,
                voxel_size = voxel_size,
                point_cloud_range = point_cloud_range, 
                num_filters = [32],
                # init_cfg = dict(
                #     type = 'Pretrained',
                #     checkpoint = '/mnt/afs2/ppshen/densegop/tools/checkpoints/GAC-3M1-sw13.0-ch32-vfe_PCSeg.pth',
                # )
            ),
            map_to_bev = dict(
                type = 'PointPillarScatter_Seg',
                num_bev_features = 32,
                grid_size = grid_size,
                batch_size = batch_size,
            ),
            backbone_2d_pvb = dict(
                type = 'BaseBEVBackbone_FPN',
                downsample_input = None,
                use_deconv = True,
                use_ca = True,
                input_channels = 32,    # num_bev_features
                layer_nums = [4, 4, 4],
                layer_strides = [2, 2, 2],
                num_filters = [80, 96, 128],
                upsample_strides = [1, 2, 4],
                num_upsample_filters = [32, 32, 32],
                out_indices = [0, 1, 2],
                return_list = ['cat'],
                task_name = 'PVB',
            ),
            # dense_head_det_fusion_pvb = dict(
            #     type = 'pp_heavy_head',
            #     # class_agnostic = False,
            #     cfg = dict(
            #         class_names = class_names,
            #         task = None,
            #         to_caffe = False,
            #         set_quantization = False,
            #         rpn_head_group_nums = [1, 1, 1],
            #         use_ppheavy_loss = True,
            #         task_name = 'PVB',
            #         box_coder = 'ResidualCoder',
            #         rpn_stage = dict(
            #             type = 'PointPillarsScatter',
            #             fixed = False,
            #             downsample_factor = 2,
            #             num_head_input_features = 32,
            #             encode_bg_as_zeros = True,
            #             use_sigmoid_score = True,
            #             use_rotated_nms = True,
            #             autoweight = False,
            #             auto_only_ped_cyc = True,
            #             auto_only_ped_cyc_truck = False,
            #             autoweight_coff = 1.0,
            #             autoweight_init = True,
            #             auto_pos_cls = False,
            #             autoweight_class_bal = True,
            #             autoweight_class_bal_coff = 0.2,
            #             autoweight_fixed = False,
            #             bn_mom = -1.0,
            #             ghm_cls = False,
            #             ghm_cls_bins = 30,
            #             ghm_cls_mom = 0.75,
            #             ghm_reg = False,
            #             ghm_reg_bins = 10,
            #             ghm_reg_mom = 0.7,
            #             autoloss = False,
            #             norm_reg_encod = False,
            #             eql_cls = False,
            #             eql_rare_class = [],
            #             ohnm = False,
            #             ohnm_random = False,
            #             ohnm_frac = 0.0,
            #             focal_loss = True,
            #             non_focal_loss_weight = False,
            #             gfl = dict(
            #                 use_qfl = False,
            #             ),
            #             backbone = dict(
            #                 part_reg_enabled = True,
            #                 pre_activation = False,
            #                 use_groupnorm = False,
            #                 use_bnet = False,
            #                 bnet_k = 3,
            #                 vov_trans_concat = False,
            #                 vov_Se = False,
            #                 vov_idmap = False,
            #                 repvgg = False,
            #                 keep_res = False,
            #                 se_reduction = 16,
            #                 head_se = False,
            #                 vfe_se = False,
            #                 backbone_se = False,
            #                 fix_backbone = False,
            #                 fpn_se = False,
            #                 ghost_module_use = False,
            #                 ghost_module_layer_range = [],
            #                 asymmres_module_use = False,
            #                 convgroups = [1, 1, 1],
            #                 rfbblock_use = False,
            #                 rfbblock_scale = 0.1,
            #                 num_groups = 16,
            #                 fpn = dict(
            #                     pp_fpn = False,
            #                     pp_fpn_heavy = True,
            #                     fpga_fpnv1 = False,
            #                     fpga_fpnv2 = False,
            #                     fpn_stack_num = 1,
            #                     pp_fpn_heavy_upsample = False,
            #                     pp_fpn_ori = False,
            #                     fconv3 = False,
            #                     fpn_out2 = False,
            #                     fpn_sum = True,
            #                     fpn_concat = False,
            #                 ),
            #             ),
            #             iou_head = dict(
            #                 use = False,
            #                 use_autoweight = True,
            #                 auto_only_ped_cyc = True,
            #                 auto_only_ped_cyc_truck = False,
            #                 iou_weight = 1.0,
            #                 autoweight_coff = 1.0,
            #                 autoweight_class_bal_coff = 0.2,
            #                 label_no_grad = True,
            #                 rotated_bev_iou = False,
            #                 rotated_3d_iou = False,
            #             ),
            #             rpn_head = dict(
            #                 type = 'MultiHeadRPN',
            #                 layer_strides = [1, 2],
            #                 upsample_strides = [1, 2],
            #                 layer_nums = [5, 5],
            #                 num_filters = [128, 256],
            #                 num_upsample_filters = [ 256, 256],
            #                 use_groupnorm = False,
            #                 num_groups = 32,
            #                 num_input_features = 32,
            #                 use_direction_classifier = True,
            #                 encode_rad_error_by_sin = True,
            #                 head_init_weight = True,
            #                 loc_sigma = 4.0,
            #                 reg_variance = dict(
            #                     output = True,
            #                     kl_loss = True,
            #                 ),
            #                 reg_attributes_weights = dict(
            #                     use = False,
            #                     bev = 1.0,
            #                     h3d = 1.0,
            #                 ),
            #                 fpga_iou_loss = dict(
            #                     use = True,
            #                     cls_iou_loss = True,
            #                     loc_iou_loss = False,
            #                 ),
            #                 rpn_base_args = dict(
            #                     split_fpn = 0,
            #                     use_norm = True,
            #                     layer_nums = [4, 5, 6],
            #                     layer_strides = [2, 2, 2],
            #                     num_filters = [128, 160, 192],
            #                     upsample_strides = [1, 2, 4],
            #                     num_upsample_filters = [32, 32, 32],
            #                     num_input_features = 32,
            #                     use_groupnorm = False,
            #                     num_groups = 16,
            #                     use_deconv = True,
            #                     pre_conv = True,
            #                     resblock = 'BasicBlock',
            #                     res_stage_1_split_conv = True,
            #                     num_res_stages = 3,
            #                     num_res_blocks = [5, 15, 16],
            #                     res_stage_dilation = [False, False, False],
            #                     num_res_filters = [160, 192, 224],
            #                     groups = 1,
            #                     width_per_group = 64,
            #                     zero_init_residual = False,
            #                     droprate = 0.0,
            #                     num_hourglass_stacks = 1,
            #                     num_hourglass_blocks = 1,
            #                     num_hourglass_feats = 128,
            #                     dense_layer_type = 'B',
            #                     num_dense_stages = 3,
            #                     num_dense_blocks = [4, 5, 6],
            #                     num_dense_filters = [128, 64, 64]
            #                 ),
            #                 loss_weights = dict(
            #                     cls_weight = 1.0,
            #                     loc_weight = 2.0,
            #                     dir_weight = 0.2
            #                 ),
            #                 rpn_head_args = [
            #                     dict(
            #                         input_channels = pts_in_channels,
            #                         class_name = class_names,
            #                         num_filters = pts_in_channels,
            #                         conv_nums = 0,
            #                         dilation = [],
            #                         use_se = False,
            #                         use_res = False,
            #                         use_glore = False,
            #                         use_shuffle = False,
            #                         downsample_level = 2,
            #                         ratio = 4,
            #                         se_ratio = 8,
            #                         num_res = 1,
            #                         ds_keep_ratio = 1.0,
            #                         ks1 = 3
            #                     ),
            #                 ]
            #             ),
            #         ),
            #         fpga_nms = True,
            #         train = dict(
            #             split = 'train',
            #             ls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            #             pos_ls = 4.0,
            #         ),
            #         test = dict(
            #             split = 'val',
            #             post_center_limit_range = point_cloud_range,
            #             nms_thresh = 0.1,
            #             nms_thresh_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            #             score_thresh_list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            #             eval_in_lidar = True,
            #             eval_with_closebox = False,
            #             max_num = 100
            #         ),
            #     ),
            # ),
            post_processing = dict(
                recall_thresh_list = [0.3, 0.5, 0.7],
                eval_metric = 'kitti',
            ),
        ),
    ),
    # multimodal_gop=dict(
    #     type="PointPillars",
    #     downsample_input = None,
    #     use_deconv = True,
    #     use_ca = False,
    #     input_channels = 32,    # num_bev_features
    #     layer_nums = [4, 4, 4],
    #     layer_strides = [2, 2, 2],
    #     num_filters = [80, 96, 128],
    #     upsample_strides = [1, 2, 4],
    #     num_upsample_filters = [32, 32, 32],
    #     out_indices = [0, 1, 2],
    #     return_list = ['cat'],
    #     task_name = 'GOP_DRIVING',    # default None
    #     # vfe = dict(
    #     #     type = 'PillarVFE_Seg',
    #     #     with_distance = False,
    #     #     # use_abslote_xyz = True,
    #     #     use_norm = True,
    #     #     num_point_features = 4,
    #     #     voxel_size = gop_voxel_size,
    #     #     point_cloud_range = gop_point_cloud_range, 
    #     #     num_filters = [32],
    #     #     init_cfg = dict(
    #     #         type = 'Pretrained',
    #     #         checkpoint = '/iag_ad_01/ad/ppshen/densegop/tools/checkpoints/GAC-3M1-sw13.0-ch32-vfe_PCSeg.pth',
    #     #     )
    #     # ),
    #     # map_to_bev = dict(
    #     #     type = 'PointPillarScatter_Seg',
    #     #     num_bev_features = 32,
    #     #     grid_size = gop_grid_size,
    #     #     mask_x_coord = abs((point_cloud_range[0] - gop_point_cloud_range[0]) / voxel_size[0]),
    #     #     mask_y_coord = abs((point_cloud_range[1] - gop_point_cloud_range[1]) / voxel_size[1]),
    #     # ),
    #     multimodal_head=dict(
    #         type='MultiModal_PVB_GOP_HEAD',
    #         model_cfg=dict(
    #             # class_names = gop_class_names,
    #             camera_names=["center_camera_fov30", "center_camera_fov120"],
    #             multi_task_names = ['GOP_DRIVING', ],    # ['PVB_DRIVING', 'GOP_DRIVING', 'PVB_PARKING', 'GOP_PARKING']
    #             grid_size = gop_grid_size,
    #             voxel_size = gop_voxel_size,
    #             point_cloud_range = gop_point_cloud_range,
    #             num_rawpoint_features = 4,
    #             num_point_features = 32,
    #             depth_downsample_factor = None,
    #             fuser_later = True,
    #             freeze = dict(
    #                 # fix_modules = ['image_backbone', 'neck', 'vtransform', 'backbone_2d_cam_pvb_parking', 'backbone_2d_cam_gop_parking',
    #                 #                'vfe', 'map_to_bev_module', 'backbone_2d_pvb_parking', 'backbone_2d_gop_parking'],
    #                 fix_modules = None,
    #                 ignore_keys = None,
    #                 grad_back_flag = False,
    #             ),
    #             # neck=dict(
    #             #     type='CustomFPN',
    #             #     used_cams=gop_cams,
    #             #     all_task_cams=cams,
    #             #     in_channels=[256, 256, 256, 256],   # [128, 256, 512]
    #             #     out_channels=256,
    #             #     start_level=0,
    #             #     end_level=-1,
    #             #     num_outs=1,
    #             #     out_ids=[0]),
    #             vtransform = dict(
    #                 type = 'AttentionTransform_Lidaraug_Update',
    #                 model_cfg = dict(
    #                     num_view = len(gop_cams),
    #                     image_size = gop_image_size,
    #                     used_cams=gop_cams,
    #                     all_task_cams=cams,
    #                     in_channel = 256,
    #                     out_channel = 256,
    #                     outp_proj = dict(
    #                         mid_channel = 64,   # replace OUT_CHANNEL for gridsample
    #                         final_out_channel = 256,
    #                     ),
    #                     num_points_in_pillar = 15,
    #                     num_feature_levels = 1,
    #                     use_cams_embeds = True,
    #                     feature_size = [64, 128],
    #                     xbound = [0, 102.4, 0.64],
    #                     ybound = [-7.68, 7.68, 0.64],
    #                     zbound = [-4.0, 4.0, 0.2],
    #                 ),
    #             ),
    #             backbone_2d_cam_gop_driving = dict(
    #                 type = 'BEVImageBackbone_FPN',
    #                 input_bev_size = [64, 96],  # bev-map (h, w), [64, 160] -> [64, 64] for deploy
    #                 downsample_input = dict(
    #                     stride = 0.5,
    #                     in_channel = 256,
    #                     out_channel = 64,
    #                 ),
    #                 use_deconv = True,
    #                 use_ca = True,
    #                 input_channels = 64,    # num_bev_features
    #                 layer_nums = [4, 4, 4],
    #                 layer_strides = [1, 2, 2],
    #                 num_filters = [80, 96, 128],
    #                 upsample_strides = [1, 2, 4],
    #                 num_upsample_filters = [32, 32, 32],
    #                 out_indices = [0, 1, 2],
    #                 return_list = ['cat'],
    #                 task_name = 'GOP_DRIVING',
    #             ),
    #             fuser_gop_driving = dict(
    #                 type = 'ConvFuser_later_atten',
    #                 in_channel = 192,
    #                 out_channel = 96,
    #                 layers_conv = 3,
    #                 task_name = 'GOP_DRIVING',
    #             ),
    #             dense_head_det_fusion_gop_driving = dict(
    #                 type = 'pp_heavy_head_v2',
    #                 # class_agnostic = False,
    #                 cfg = dict(
    #                     class_names = gop_class_names,
    #                     task = 'fusion',
    #                     to_caffe = False, set_quantization = False,
    #                     rpn_head_group_nums = [1, 1, 1],
    #                     use_ppheavy_loss = True,
    #                     task_name = 'GOP_DRIVING',
    #                     box_coder = 'ResidualCoder',
    #                     rpn_stage = dict(
    #                         type = 'PointPillarsScatter',
    #                         fixed = False,
    #                         downsample_factor = 2,
    #                         num_head_input_features = 32,
    #                         encode_bg_as_zeros = True,
    #                         use_sigmoid_score = True,
    #                         use_rotated_nms = True,
    #                         autoweight = False, autoweight_coff = 1.0, autoweight_init = True, autoweight_fixed = False,
    #                         auto_only_ped_cyc = True, auto_only_ped_cyc_truck = False,
    #                         auto_pos_cls = False, autoweight_class_bal = True, autoweight_class_bal_coff = 0.2,
    #                         bn_mom = -1.0,
    #                         ghm_cls = False, ghm_cls_bins = 30, ghm_cls_mom = 0.75,
    #                         ghm_reg = False, ghm_reg_bins = 10, ghm_reg_mom = 0.7,
    #                         autoloss = False, norm_reg_encod = False,
    #                         eql_cls = False, eql_rare_class = [],
    #                         ohnm = False, ohnm_random = False, ohnm_frac = 0.0,
    #                         focal_loss = True, non_focal_loss_weight = False,
    #                         gfl = dict(use_qfl = False,),
    #                         backbone = dict(
    #                             part_reg_enabled = True, pre_activation = False,
    #                             use_groupnorm = False, use_bnet = False, bnet_k = 3,
    #                             vov_trans_concat = False, vov_Se = False, vov_idmap = False,
    #                             repvgg = False, keep_res = False, fix_backbone = False,
    #                             se_reduction = 16, head_se = False, vfe_se = False, backbone_se = False, fpn_se = False,
    #                             ghost_module_use = False, ghost_module_layer_range = [],
    #                             asymmres_module_use = False, convgroups = [1, 1, 1], num_groups = 16,
    #                             rfbblock_use = False, rfbblock_scale = 0.1,
    #                             fpn = dict(
    #                                 pp_fpn = False, pp_fpn_heavy = True,
    #                                 fpga_fpnv1 = False, fpga_fpnv2 = False, fpn_stack_num = 1,
    #                                 pp_fpn_heavy_upsample = False, pp_fpn_ori = False,
    #                                 fconv3 = False, fpn_out2 = False, fpn_sum = True, fpn_concat = False,
    #                             ),
    #                         ),
    #                         iou_head = dict(
    #                             use = False,
    #                             use_autoweight = True, auto_only_ped_cyc = True, auto_only_ped_cyc_truck = False,
    #                             iou_weight = 1.0, autoweight_coff = 1.0, autoweight_class_bal_coff = 0.2,
    #                             label_no_grad = True, rotated_bev_iou = False, rotated_3d_iou = False,
    #                         ),
    #                         rpn_head = dict(
    #                             type = 'MultiHeadRPN',
    #                             layer_strides = [1, 2], upsample_strides = [1, 2], layer_nums = [5, 5],
    #                             num_filters = [128, 256], num_upsample_filters = [ 256, 256],
    #                             use_groupnorm = False, num_groups = 32, num_input_features = 32,
    #                             use_direction_classifier = True, encode_rad_error_by_sin = True,
    #                             head_init_weight = True, loc_sigma = 4.0,
    #                             reg_variance = dict(output = True, kl_loss = True,),
    #                             reg_attributes_weights = dict(use = False, bev = 1.0, h3d = 1.0,),
    #                             fpga_iou_loss = dict(use = True, cls_iou_loss = True, loc_iou_loss = False,),
    #                             rpn_base_args = dict(
    #                                 split_fpn = 0, use_norm = True,
    #                                 layer_nums = [4, 5, 6], layer_strides = [2, 2, 2],
    #                                 num_filters = [128, 160, 192], upsample_strides = [1, 2, 4], num_upsample_filters = [32, 32, 32],
    #                                 num_input_features = 32,
    #                                 use_groupnorm = False, num_groups = 16, use_deconv = True, pre_conv = True,
    #                                 resblock = 'BasicBlock',
    #                                 res_stage_1_split_conv = True, num_res_stages = 3, num_res_blocks = [5, 15, 16],
    #                                 res_stage_dilation = [False, False, False],
    #                                 num_res_filters = [160, 192, 224],
    #                                 groups = 1, width_per_group = 64,
    #                                 zero_init_residual = False, droprate = 0.0,
    #                                 num_hourglass_stacks = 1, num_hourglass_blocks = 1, num_hourglass_feats = 128,
    #                                 dense_layer_type = 'B',
    #                                 num_dense_stages = 3, num_dense_blocks = [4, 5, 6], num_dense_filters = [128, 64, 64]
    #                             ),
    #                             loss_weights = dict(cls_weight = 1.0, loc_weight = 2.0, dir_weight = 0.2),
    #                             rpn_head_args = [
    #                                 dict(
    #                                     input_channels = 96,
    #                                     class_name = gop_class_names,
    #                                     num_filters = 96, conv_nums = 0, dilation = [],
    #                                     use_se = False, use_res = False, use_glore = False, use_shuffle = False,
    #                                     downsample_level = 2, ratio = 4, se_ratio = 8,
    #                                     num_res = 1, ds_keep_ratio = 1.0, ks1 = 3
    #                                 ),
    #                             ]
    #                         ),
    #                     ),
    #                     fpga_nms = True,
    #                     train = dict(
    #                         split = 'train', pos_ls = 4.0,
    #                         ls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                     ),
    #                     test = dict(
    #                         split = 'val',
    #                         post_center_limit_range = gop_point_cloud_range,
    #                         nms_thresh = 0.1,
    #                         nms_thresh_list = [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                         score_thresh_list =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #                         eval_in_lidar = True, eval_with_closebox = False,
    #                     ),
    #                 ),
    #             ),
    #         ),
    #     ),
    # ),
    # pts_query_generator=dict(
    #     type='PointCloudQueryGenerator',
    #     in_channels=pts_in_channels,
    #     hidden_channel=pts_in_channels,
    #     pts_use_cat=False,
    #     dataset='argov2',
    # ),
    # pts_bbox_head=dict(
    #     type='FarHead',
    #     use_lidar_pts_query=True,
    #     pts_in_channels=pts_in_channels,
    #     transformer_use_lidar_ca=transformer_use_lidar_ca,
    #     num_classes=len(class_names),
    #     in_channels=256,
    #     num_query=256,
    #     memory_len=512,
    #     topk_proposals=128,
    #     num_propagated=128,
    #     scalar=8, ##noise groups
    #     noise_scale=1.0, 
    #     dn_weight=1.0, ##dn loss weight
    #     split=0.75, ###positive rate
    #     offset=0.5,
    #     offset_p=0.0,
    #     num_smp_per_gt=3,
    #     with_dn=True,
    #     with_ego_pos=True,
    #     add_query_from_2d=False,
    #     pred_box_var=False,  # note add box uncertainty
    #     train_use_gt_depth=False,
    #     val_use_gt_depth=False,
    #     add_multi_depth_proposal=True,
    #     multi_depth_config={'topk': 1, 'range_min': 30,},  # 'bin_unit': 1, 'step_num': 4,
    #     return_bbox2d_scores=True,
    #     return_context_feat=True,
    #     code_size=10,
    #     code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    #     transformer=dict(
    #         type='Far3DTransformer',
    #         decoder=dict(
    #             type='Far3DTransformerDecoder',
    #             embed_dims=256,
    #             num_layers=4,
    #             transformerlayers=dict(
    #                 type='Far3DTemporalDecoderLayer',
    #                 batch_first=True,
    #                 attn_cfgs=[
    #                     dict(
    #                         type='MultiheadAttention',
    #                         embed_dims=256,
    #                         num_heads=8,
    #                         dropout=0.1),
    #                     dict(
    #                         type='Far3DMixedCrossAttention',
    #                         embed_dims=256,
    #                         num_groups=8,
    #                         num_levels=4,
    #                         num_cams=num_cams,
    #                         dropout=0.1,
    #                         num_pts=13,
    #                         bias=2.,
    #                         use_relu6=True,
    #                         attn_cfg=dict(
    #                             type='MultiheadAttention',
    #                             batch_first=True,
    #                             embed_dims=256,
    #                             num_heads=8,
    #                             dropout=0.1) if transformer_use_lidar_ca else None,),
    #                     ],
    #                 feedforward_channels=2048,
    #                 ffn_dropout=0.1,
    #                 with_cp=False,  ###use checkpoint to save memory
    #                 operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
    #                                  'ffn', 'norm')),
    #         )),
    #     bbox_coder=dict(
    #         type='NMSFreeCoder2',
    #         post_center_range=extend_bbox_pred_range,
    #         pc_range=bbox_pred_range,
    #         max_num=300,
    #         voxel_size=voxel_size,
    #         num_classes=len(class_names)), 
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=2.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    #     loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    # ),
    
    lc_fusion=dict(
        type='LCFusion',
        lic = 96,
        imc = 256,
    ),
    bev_encoder=dict(
            type='BevEncoder',
            num_cams=num_cams,
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            modality='fusion',
            lidar_dropout_prob=0.1,
            lidar_spatial_rate=0.2,
            lidar_mask_ratio=0.2,
            fuser= dict(
                type='LCFusionV2',
                with_reduce_conv=True,
                bev_h=bev_h_,
                bev_w=bev_w_,
                lic = 96,
                imc = 256,
            ),
            # streaming_cfg=dict(
            #     streaming_bev=True,
            #     batch_size=batch_size,
            #     fusion_cfg=dict(
            #         type='ConvGRU',
            #         out_channels=_dim_,
            #     )
            # ),
            # bev_backbone=dict(
            #     type='ResNetBev',
            #     numC_input=_dim_,
            #     num_layer=[1, 1, 1, 1],
            #     num_channels=[256, 320, 384, 512],
            #     stride=[1,2,2,2],
            #     # block_type='Basic',
            #     block_type='BottleNeck',
            #     norm_cfg=dict(
            #         type='SyncBN',
            #         eps=0.001,
            #         momentum=0.01
            #     )
            # ),
            bev_backbone=dict(
                type='ConvNeXt',
                inplanes=_dim_,
                depths=[3, 3, 9, 3],
                outplanes=[64, 128, 192, 256],
                # outplanes=[128, 256, 512, 1024],
                drop_path_rate=0.2,
                layer_scale_init_value=1.e-6,
                freeze_layers=[], # layer idx start from 0
                ActType='ReLU',
                downsample_strides=[1, 2, 2, 2],
                downsample_kernels=[3, 3, 3, 3],
                fusion_bev_supervision=False,
                # normalize=dict(type='SyncBN',)
                normalize=dict(type='BN',)
            ),
            bev_sneck=dict(
                type='SECONDTransFPNV3',
                inplanes=[64, 128, 192, 256],
                outplanes=[[256, 256, 256, 256]],
                upsample='bilinear',
                upsample_strides=[[1, 2, 4, 8]],
                align_corners=False,
                conv_bias=False,
                use_conv_for_no_stride=True,
                attention=False,
                # normalize=dict(type='SyncBN',eps=0.001,momentum=0.01)
                normalize=dict(type='BN',eps=0.001,momentum=0.01)
            ),
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=map_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            num_cams=num_cams,
                            pc_range=map_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                im2col_step=128,
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
                ),
            ),
    lane3d_head=dict(
        type='MapTRv2HeadDecoder',
        types_dict=lane3d_types,
        colors_dict=lane3d_colors,
        bev_h=bev_h_,
        bev_w=bev_w_,
        # num_query=900, # no used
        num_vec_one2one=map_num_vec,
        # num_vec_one2many=300, # close one2many
        # k_one2many=3, # close one2many
        num_pts_per_vec=map_fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance',
        transform_method='minmax',
        # gt_shift_pts_pattern='v6',
        gt_shift_pts_pattern=gt_shift_pts_pattern,
        seperate_query=True,
        sep_classes_ratio=[0.5, 0.25, 0.1, 0.15],
        ignore_cfg={5: [1], 7: [1], 9: [1]}, # vd_all_intersection: 5, roadside: 1
        num_classes=map_num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        use_geo_loss=True,
        use_close_weight=True,
        close_weight_value=2.5, #zyq:5.0->2.5
        aux_seg=aux_seg_cfg,
        pts_uncertain=True,
        # z_cfg=z_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformerDecoder',
            num_cams=num_cams,
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            embed_dims=_dim_,
            pc_range=map_range,
            decoder=dict(
                type='MapTRDecoder',
                num_layers=4,
                return_intermediate=True,
                query_pos_embedding='instance',
                num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0
                        ),
                        dict(
                            type='InstancePointAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
                            dropout=0,
                            output_norm=True,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type="MapNMSFreeCoder",
            # post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],     # TODO check if reasonable
            # post_center_range=[-99, -49, -99, -49, 99, 49, 99, 49],     # TODO check if reasonable
            score_threshold=0.35,
            pc_range=map_range,
            max_num=map_num_vec,
            voxel_size=voxel_size,
            num_classes=map_num_classes,
            types_dict=lane3d_types,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.3,
            loss_weight=2.0),
        loss_map_type=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.05),
        loss_map_color=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.05),
        loss_map_geo=dict(
            type="GeometricLoss",
            loss_weight=0.05,
            intra_loss_weight=1.0,
            inter_loss_weight=1.0,
            num_ins=map_num_vec,
            num_pts=map_fixed_ptsnum_per_pred_line,
            num_classes=map_num_classes,
            pc_range=map_range,
            loss_type="l1",
        ),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_map_pts_uncertain=dict(type='LaplaceNLLLoss',
                loss_weight=0.1,
                scale=50,
        ),
        loss_map_dir=dict(type='PtsWeightDirCosLoss', loss_weight=0.05),
        loss_map_normal=dict(type='NormalPtsL1Loss', loss_weight=5.0),

        # loss_seg=dict(
        #     type='MaskFocalLoss',
        #     use_sigmoid=True,
        #     loss_weight=10.0,
        # ),
        # loss_dice=dict(
        #     type='MaskDiceLoss',
        #     loss_weight=1.0,
        # ),
        # loss_pv_seg=dict(type='SimpleLoss', 
        #             pos_weight=9.6,
        #             loss_weight=2.0),
        # loss_rendered_mask=dict(type='RenderedMaskDiceLoss', weight=10.0),
    ),
    # occ3d_head=dict(
    #     type="BEVOCCHead2D_v2_new",
    #     loss_weight=1.0,
    #     in_dim=256,
    #     out_dim=256,
    #     Dz=occ_size[-1],
    #     use_mask=False,
    #     num_classes=len(occ_class_names),
    #     class_free=occ_class_names.index("free"),
    #     class_ground=occ_class_names.index("ground"),
    #     class_others_occ=occ_class_names.index("others"),
    #     class_obj=occ_class_names.index("obj"),
    #     with_ground=True,
    #     semantic=True,
    #     use_sem_geo_loss=True,
    #     use_focal_loss=False,
    #     use_dynamic_weight=False,
    #     class_occ_weights=class_weights_occ,
    #     occ_score_threshold=0.7,
    #     seg_score_threshold=0.3,
    #     with_obj=True,
    #     use_camera_mask=False,
    #     panelty_camera_mask=True,
    #     use_border_weight=True,
    #     cal_bev_loss=True,
    #     use_muti_height_weights=True,
    #     with_crop=True,
    # ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=bbox_pred_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                bbox_order="xyzwlh",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=bbox_pred_range,
            ),
            map_assigner=dict(
                type="MapHungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=10.0),
                reg_cost=dict(type="BBoxL1Cost", weight=0.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=0.0),
                # pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pts_cost=dict(type="OrderedPtsL1Cost", weight=5.0),
                score_threshold=0.35,
                pc_range=map_range,
            )
        )
    ),
)

# data pipeline
import os
conf_path = "{}/petreloss.conf".format(os.getcwd())
from path_mapping import PATH_MAPPING
file_client_args = dict(
    backend="disk"
)
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)  # we use nuimages pretrain for 2D detector
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = {
    "cams": cams,
    "Ncams": len(cams),
    "src_size": (576, 1024),
    # Image Augmentation
    "resize": (-0.05, 0.05), #zyq 0.15->0.05
    "rot": (0.0, 0.0),
    "flip": False,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.0,
}

dataset_type = 'InternalDatasetTrackStream'
data_root = "iaginfra:s3://iaginfra-v/vd_data/"

reprojection_keys = ["gt_bboxes_2d", "gt_2d_to_3d_idx", "gt_labels_2d", "gt_cam_idx"]
mono_keys = ['gt_centers_2d', 'gt_depths_cam', 'gt_bboxes_3d_cam']
pp_heavy_keys = ['voxels', 'voxel_coords', 'voxel_num_points', 'anchors_pvb', 'box_labels_pvb', 'reg_targets_pvb', 'reg_weights_pvb']
pp_heavy_dataset_cfg = dict(
    point_cloud_range = point_cloud_range,
    voxel_size = voxel_size,
    pp_heavy_names = ['PP_HEAVY_PVB', 'PP_HEAVY_GOP_DRIVING'],
    PP_HEAVY_PVB = dict(
        voxel_size = voxel_size,
        point_cloud_range = point_cloud_range,
        target_assigner = dict(
            anchor_generator = [
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[4.7, 2.1, 1.6]], # l, w, h for VEHICLE_CAR
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.6,
                    'unmatched_threshold': 0.45,
                    'feature_map_scale_factor': 1,
                    'class_name': 'VEHICLE_CAR'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[6.9, 2.5, 2.8]], # l, w, h for VEHICLE_TRUCK
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.55,
                    'unmatched_threshold': 0.4,
                    'feature_map_scale_factor': 1,
                    'class_name': 'VEHICLE_TRUCK'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[12.0, 2.6, 3.5]], # l, w, h for VEHICLE_BUS
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.55,
                    'unmatched_threshold': 0.4,
                    'feature_map_scale_factor': 1,
                    'class_name': 'VEHICLE_BUS'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[6.0, 2.5, 2.5]], # l, w, h for SPECIAL_VEHICLE
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.5,
                    'unmatched_threshold': 0.35,
                    'feature_map_scale_factor': 1,
                    'class_name': 'SPECIAL_VEHICLE'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[2.0, 0.9, 1.6]], # l, w, h for VEHICLE_TRIKE
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.5,
                    'unmatched_threshold': 0.35,
                    'feature_map_scale_factor': 1,
                    'class_name': 'VEHICLE_TRIKE'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[1.8, 0.7, 1.5]], # l, w, h for BIKE_BICYCLE
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.5,
                    'unmatched_threshold': 0.35,
                    'feature_map_scale_factor': 1,
                    'class_name': 'BIKE_BICYCLE'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': point_cloud_range,
                    'sizes': [[0.8, 0.8, 1.8]], # l, w, h for PEDESTRIAN
                    'strides': [voxel_size[0] * pp_heavy_downsample_factor, voxel_size[1] * pp_heavy_downsample_factor, 0.0], 
                    'offsets': pp_heavy_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.5,
                    'unmatched_threshold': 0.35,
                    'feature_map_scale_factor': 1,
                    'class_name': 'PEDESTRIAN'
                }
            ],
            region_similarity_fn = 'nearest_iou_similarity',
            sample_pos_fraction = -1.0,
            sample_size = 512,
        ),
        box_coder = 'ResidualCoder',
        downsample_factor = pp_heavy_downsample_factor,
    ),
    PP_HEAVY_GOP_DRIVING = dict(
        voxel_size = voxel_size,
        point_cloud_range = gop_point_cloud_range,
        target_assigner = dict(
            anchor_generator = [
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[0.51, 0.49, 0.94]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.25,
                    'unmatched_threshold': 0.1,
                    'feature_map_scale_factor': 1,
                    'class_name': 'CONE'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[0.42, 0.39, 0.88]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.25,
                    'unmatched_threshold': 0.1,
                    'feature_map_scale_factor': 1,
                    'class_name': 'POLE'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[0.80, 0.78, 0.96]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.4,
                    'unmatched_threshold': 0.25,
                    'feature_map_scale_factor': 1,
                    'class_name': 'ISOLATION_BARREL'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[0.82, 0.62, 0.62]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.25,
                    'unmatched_threshold': 0.1,
                    'feature_map_scale_factor': 1,
                    'class_name': 'TRIANGLE_WARNING'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[0.55, 0.51, 0.47]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.25,
                    'unmatched_threshold': 0.1,
                    'feature_map_scale_factor': 1,
                    'class_name': 'ANIMAL'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[3.02, 0.46, 1.22]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets':gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.4,
                    'unmatched_threshold': 0.25,
                    'feature_map_scale_factor': 1,
                    'class_name': 'GATE_ROD'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[1.51, 0.62, 0.88]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.4,
                    'unmatched_threshold': 0.25,
                    'feature_map_scale_factor': 1,
                    'class_name': 'BARRIER'
                },
                {
                    'type': "anchor_generator_stride",
                    'anchor_range': gop_point_cloud_range,
                    'sizes': [[1.43, 0.64, 1.29]],
                    'strides': [0.32, 0.32, 0.0], 
                    'offsets': gop_anchor_offsets,
                    'rotations': [0, 1.57],
                    'matched_threshold': 0.4,
                    'unmatched_threshold': 0.25,
                    'feature_map_scale_factor': 1,
                    'class_name': 'CONSTRUCTION_SIGN'
                },
            ],
            region_similarity_fn = 'nearest_iou_similarity',
            sample_pos_fraction = -1.0,
            sample_size = 512,
        ),
        box_coder = 'ResidualCoder',
        downsample_factor = pp_heavy_downsample_factor,
    )
)

laneline_keys = ["map_gt_labels_3d", "map_gt_bboxes_3d", "gt_seg_mask", "gt_seg_offset", "gt_seg_type", "gt_seg_color",  "map_gt_shifts_pts_list", "map_gt_pts_types_list", "map_gt_pts_colors_list"]
occ_keys = [
    "occ_voxel_semantics",
    "occ_voxel_instances",
    "occ_camera_masks",
]

train_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args,
        lidar_range=[119.706, 21, 3.773059] # HFOV, VFOV, blind dist
    ),
    # dict(
    #     type='PointsRotTransDisturbance',
    #     rot_range=[-0.087, 0.087],
    #     translation_std=[0.5, 0.5, 0],
    #     p=0.3
    # ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     file_client_args=file_client_args),
    dict(
        type='LoadMultiViewImageFromFilesSenseWarp',
        file_client_args=file_client_args,
    ),
    dict(
        type='VCRemapObjAnnos',
        convert_2d_center=True,
    ),
    dict(
        type="LoadAnnotations2D3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_mono_rpn=True,
        with_plan=False,
        with_map_3d=True,
        with_occ_3d=True,
        with_obj_traj_pred=False,
    ),
    dict(type="ObjectRangeFilter2D3D", point_cloud_range=bbox_pred_range),
    dict(type="ObjectNameFilter2D3D", classes={'pvb': class_names, 'gop': gop_class_names}),
    dict(
        type="ResizeCropFlipRotImageOnlyResize2D3D",
        data_config=ida_aug_conf,
        apply_ida=True,
    ),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='PhotoMetricDistortionMultiViewImage',
        brightness_delta=12,
        contrast_range=(0.1, 1.1),
        saturation_range=(0.1, 1.1),
        hue_delta=5,
        ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='TransPointsToVoxels', point_cloud_range=point_cloud_range, num_point_features=4, voxel_size=voxel_size, pillar_label=True),
    dict(type='PreparePPHeavyGT', is_training=True, class_names={'pvb': class_names, 'gop': gop_class_names}, pp_heavy_names=['PP_HEAVY_PVB', 'PP_HEAVY_GOP_DRIVING'], dataset_cfg=pp_heavy_dataset_cfg),
    dict(type="NormalizeMultiviewImageSpeedUp", img_norm_cfg=img_norm_cfg),
    dict(type="PadMultiViewImageSpeedUp", size_divisor=32),
    dict(
        type="VectorizeLocalMap",
        fixed_num=20,
        padding_value=-10000,
        map_range=map_range,
        range_filter_flag=True,
        types = lane3d_types,
        colors = lane3d_colors,
        vd_type_mp = {'LANELINE': 0, 'ROADSIDE': 1, 'CROSSWALK': 2, 'STOPLINE': 3},
        aux_seg=aux_seg_cfg,
        bev_h=seg_bev_h_,
        bev_w=seg_bev_w_,
        use_multi_class=True,
        fillPoly=True,
        filter_unclear=False,
    ),
    dict(
        type="GenerateMapGTShifts",
        pattern=gt_shift_pts_pattern,
        src_key="map_gt_bboxes_3d",
        pts_key="map_gt_shifts_pts_list",
        type_key="map_gt_pts_types_list",
        color_key="map_gt_pts_colors_list",
    ),
    dict(type="LoadOccupancyMapAnnotations3Dv2",
         with_gt_occ3d=True,
         file_client_args=file_client_args,
         need_mask_occ_areas=False,
         adjust_coords=True,
         class_free=occ_class_names.index("free"),
         class_ground=occ_class_names.index("ground"),
         class_others_occ=occ_class_names.index("others"),
         class_obj = occ_class_names.index("obj"),
         collapse_z=False,
         target_pc_range=target_occ_pc_range,
         ori_pc_range=ori_occ_pc_range,
         target_grid_shape=occ_size,
         ori_grid_shape=[550, 250, 18],
         ori_voxel_size = [0.2, 0.2, 0.2],
         target_voxel_size = [0.4, 0.4, 0.4],
         add_obj_dections = True,
         read_refresh_npz = True
    ),
    dict(
        type="PETRFormatBundle3D",
        class_names={'pvb': class_names, 'gop': gop_class_names},
        collect_keys=collect_keys + ["prev_exists"],
    ),
    dict(type='Collect3D', 
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'prev_exists'] 
            + collect_keys 
            + reprojection_keys 
            + mono_keys 
            + pp_heavy_keys 
            + gop_driving_keys 
            + laneline_keys
            + occ_keys
            + ["data_tag", "pred_bbox3d_range_valid"],
         meta_keys=(
             'filename', 
             'ori_shape', 
             'img_shape', 
             'pad_shape', 
             'scale_factor', 
             'flip', 
             'box_mode_3d', 
             'box_type_3d', 
             'img_norm_cfg', 
             'scene_token', 
             'gt_bboxes_3d', 
             'gt_labels_3d',
             'lidar2img',
            ), 
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=5,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     file_client_args=file_client_args),
    dict(
        type="LoadMultiViewImageFromFilesResize2",
        to_float32=True,
        file_client_args=file_client_args,
        resize_h=576,
        resize_w=1024,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TransPointsToVoxels', point_cloud_range=point_cloud_range, num_point_features=4, voxel_size=voxel_size, pillar_label=True),
    dict(type='PreparePPHeavyGT', is_training=False, class_names={'pvb': class_names, 'gop': gop_class_names}, pp_heavy_names=['PP_HEAVY_PVB', 'PP_HEAVY_GOP_DRIVING'], dataset_cfg=pp_heavy_dataset_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys + ['prev_exists'],
                class_names={'pvb': class_names, 'gop': gop_class_names},
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'img', 'prev_exists'] + collect_keys + ['voxels', 'voxel_coords', 'voxel_num_points', 'anchors_pvb', 'anchors_gop_driving'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                            'box_type_3d', 'img_norm_cfg', 'scene_token', 'lidar2img', 'intrinsics', 'extrinsics'))
        ])
]

new_test = [
    # city 888w
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=file_client_args,
        ann_file="/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/train/train_fram_list.json",
        flag_file="/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/train/train_fram_list_flag.json",
        sdmap_range=sdmap_range,
        dropout_sd_prob=0.2,
        split_json=True,
        load_obj_traj=False,
        use_sub_class=True,
        use_epai_label=True,
        is_pap_data=False,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training, 3d laneline没有自车运动信息，没法做时序训练
        seq_mode=True,  # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        cams=cams,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        # calib_config_path="/mnt/afs2/zhangxianghang/perception/sense_spider/epai8-001",
        box_type_3d="LiDAR",
        # data_tag="vd_all",  # vd_all
        data_tag="lane3d",  # vd_all
        use_pts_sdmap=True,
    ),
]

####################### road& occ finetune 用到的数据 end ########################
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train_loader=dict(pin_memory=True, prefetch_factor=3),
    train=new_test,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        file_client_args=file_client_args,
        ann_file="/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/val/val_fram_list.json",
        split_json=True,
        pipeline=test_pipeline,
        classes=class_names,
        cams=cams,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "img_metas"],
        queue_length=queue_length,
        test_mode=True,
        extrinsic_adjust_camera_name=None,
        # calib_config_path="/mnt/afs2/chenxuepan/sense_spider/epai8-001",
        calib_config_path=None,
        box_type_3d="LiDAR",
        use_pts_sdmap=True,
        loading_sdmap=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        file_client_args=file_client_args,
        ann_file='/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/val/val_fram_list.json',
        flag_file='/root/export/Datasets/ST_DATA_0226_mini/train_list/ascend/val/val_fram_list_flag.json',
        split_json=True,
        seq_mode=True,
        use_sub_class=True,
        use_epai_label=True,
        is_pap_data=True,
        pipeline=test_pipeline,
        classes=class_names,
        cams=cams,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "img_metas"],
        queue_length=queue_length,
        test_mode=True,
        # evaluator_type='GACSubtype',
        # cls_conf_threshold=0.3,
        # cls_iou_threshold=0.5,
        # point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,
        # calib_config_path="/mnt/afs2/zhangxianghang/perception/sense_spider/epai8-001",
        box_type_3d="LiDAR",
        data_tag="pvb",
        loading_sdmap=False,
        # lidar_type=['top_center_lidar', 'front_lidar']
        # lidar_type=['front_lidar', 'top_center_lidar'] # first front_lidar
    ),

    # shuffler_sampler=dict(type='InfiniteRandomDropGroupEachSampleInBatchSampler'),
    shuffler_sampler=dict(type='DistributedSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

max_lr = 5 * 1e-4
optimizer = dict(
    type='SOAP', 
    lr=max_lr,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }
    ),
    precondition_frequency=10,
    weight_decay=0.01)

# optimizer = dict(
#     type='FASTSOAP', 
#     lr=max_lr,
#     paramwise_cfg=dict(
#         custom_keys={
#             "img_backbone": dict(lr_mult=0.25 * (1-int(fix_backbone))), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
#             "pts_bbox_head": dict(lr_mult=1.0 * (1-int(fix_pvb_head))), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
#             "lane3d_head": dict(lr_mult=1.0),
#             "occ3d_head": dict(lr_mult=1.0),
#             "bev_encoder": dict(lr_mult=1.0),
#         },
#         norm_decay_mult=0,
#         bias_decay_mult=0
#     ),
#     precondition_frequency=100,
#     weight_decay=0.01)


optimizer_config = dict(
    type='Fp16OptimizerHookProtectGradNan', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(type='GradientCumulativeFp16OptimizerHook', 
#                         loss_scale='dynamic', 
#                         grad_clip=dict(max_norm=105, norm_type=2), 
#                         bucket_size_mb=25, 
#                         cumulative_iters=4
#                         )

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=100000000, pipeline=test_pipeline)
checkpoint_config = dict(interval=1000, max_keep_ckpts=10)
find_unused_parameters = True  #### when use checkpoint, find_unused_parameters must be False
runner = dict(
    type='IterBasedRunner', max_iters=30000)
load_from = ''
resume_from = None

# adela deployment config
head_deploy_config_mdc = json.dumps(
    {
        "__image__": "registry.sensetime.com/nart/nart:1.2.15-dev-acl-mdc_230808-580cc66e",
        "acl": {
            "remote_compile": True,
            "aoe_port": "8888",
            "use_aoe_tune": True,
            "fusion_switch_file": {
                "Switch": {"GraphFusion": {"MDCLayerNormONNXFusionPass": "off"}}
            },
        },
    }
)
deploy_config_mdc = json.dumps(
    {
        "__image__": "registry.sensetime.com/nart/nart:1.2.15-dev-acl-mdc_230808-580cc66e",
        "acl": {
            "remote_compile": True,
            "aoe_port": "8888",
            "use_aoe_tune": True,
            "fusion_switch_file": {
                "Switch": {"GraphFusion": {"MDCLayerNormONNXFusionPass": "off"}}
            },
            "aipp_params": {
                "center_camera_fov30": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "center_camera_fov120": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "left_front_camera": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "left_rear_camera": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "rear_camera": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "right_rear_camera": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
                "right_front_camera": {
                    "rbuv_swap_switch": img_norm_cfg["to_rgb"],
                    "input_format": "RGB888_U8",
                    "min": img_norm_cfg["mean"],
                    "std_var": img_norm_cfg["std"],
                    "enable_vpc": False,
                },
            },
        },
    }
)

deployment_cfg = dict(
    project_id=20,
    model_name="spetr",
    input_h=576,
    input_w=1024,
    output_names=[
        'pts_ref',
        "score",
        "bbox",
        "embedding",
        # "memory",
        # "pos_embed",
        # "object_queries",
        "pseudo_reference_points",
    ],
    deployment_platforms=["acl-ascend615-fp16-adc615-sdk230808","acl-ascend610-fp16-mdc610-sdk230808"],
    deploy_configs=[deploy_config_mdc, deploy_config_mdc],
    head_deploy_configs=[head_deploy_config_mdc, head_deploy_config_mdc],
)



