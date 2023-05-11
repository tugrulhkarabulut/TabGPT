from yacs.config import CfgNode as CN


_C = CN()
_C.INPUT = "./data/"
_C.OUTPUT = "./output/"
_C.MODEL = "gpt2"

_C.DATA = CN()
_C.DATA.TRAIN_DATASET = "classic_rock"
_C.DATA.TEST_DATASET = "classic_rock"
_C.DATA.LOAD_DATASET_FROM_DISK = True
_C.DATA.EXTEND_TOKENIZER = False


_C.TRANSFORMER_SOLVER = CN()
_C.TRANSFORMER_SOLVER.LR = 2e-5
_C.TRANSFORMER_SOLVER.TRAIN_BATCH_SIZE = 32
_C.TRANSFORMER_SOLVER.TEST_BATCH_SIZE = 32
_C.TRANSFORMER_SOLVER.GRAD_ACC_STEPS = 2
_C.TRANSFORMER_SOLVER.GRAD_CKPT = True
_C.TRANSFORMER_SOLVER.FP16 = True
_C.TRANSFORMER_SOLVER.EPOCHS = 10
_C.TRANSFORMER_SOLVER.WEIGHT_DECAY = 1e-2



def get_cfg_defaults():
    return _C.clone()