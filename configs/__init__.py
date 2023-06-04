from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.ROOT = ''

_C.DATASET.DATA1 = CN()
_C.DATASET.DATA1.MEANS = [0.] * 8
_C.DATASET.DATA1.STDS = [0.] * 8
_C.DATASET.DATA2 = CN()
_C.DATASET.DATA2.MEANS = [0.] * 10
_C.DATASET.DATA2.STDS = [0.] * 10
_C.DATASET.SAMPLE_NUM = [0] * 3
_C.DATASET.SAMPLE_ORDER = ''
_C.DATASET.CLASSES_INTEREST = [-1] * 6
_C.DATASET.FUSION = ''

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 0
_C.DATALOADER.NUM_WORKERS = 0

_C.MODEL = CN()
_C.MODEL.NAME = ''

_C.CRITERION = CN()
_C.CRITERION.NAME = ''

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = ''
_C.OPTIMIZER.LR = 0.
_C.OPTIMIZER.MOMENTUM = 0.
_C.OPTIMIZER.WEIGHT_DECAY = 0.

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = ''
_C.SCHEDULER.GAMMA = 0.
_C.SCHEDULER.MODE = ''
_C.SCHEDULER.FACTOR = 0.
_C.SCHEDULER.PATIENCE = 0

_C.EPOCHS = 0

CFG = _C.clone()
CFG.freeze()
