_base_ = "../../l2.9-df-for-yuexiang/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py"

runner = dict(type="IterBasedRunner", max_iters=30)
