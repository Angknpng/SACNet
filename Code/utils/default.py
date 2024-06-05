import os
import numpy as np
from yacs.config import CfgNode as CN


_C = CN()

_C.TRAIN = CN()

# _C.TRAIN.lr = 2e-4
# _C.TRAIN.lr_backbone_names = ["backbone.0"]
# _C.TRAIN.lr_backbone = 2e-5
# _C.TRAIN.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
# _C.TRAIN.lr_linear_proj_mult = 0.1
# _C.TRAIN.batch_size = 2
_C.TRAIN.weight_decay = 1e-4
# _C.TRAIN.epochs = 50
_C.TRAIN.lr_drop = 40#
# _C.TRAIN.lr_drop_epochs = None
# _C.TRAIN.clip_max_norm = 0.1
# _C.TRAIN.sgd = True
# _C.TRAIN.output_dir = "../results_DETR"
_C.TRAIN.device = "cuda"
_C.TRAIN.seed = 42
# _C.TRAIN.resume = ""
# _C.TRAIN.resume_default = True
# _C.TRAIN.pretrained = "../pretrained/resnet50-19c8e357.pth"
# _C.TRAIN.start_epoch = 0
# _C.TRAIN.eval = True
# _C.TRAIN.num_workers = 0
# _C.TRAIN.cache_mode = False