from yacs.config import CfgNode as CN


_C = CN()
_C.INPUT = "./data/"
_C.OUTPUT = "./output/"
_C.MODEL = "gpt2"
_C.RESUME_FROM_CKPT = False
_C.CKPT_PATH = ""

_C.DATA = CN()
_C.DATA.TRAIN_DATASET = "train_dataset"
_C.DATA.TEST_DATASET = "test_dataset"
_C.DATA.LOAD_DATASET_FROM_DISK = True
_C.DATA.EXTEND_TOKENIZER = False


_C.SOLVER = CN()
_C.SOLVER.LR = 2e-5
_C.SOLVER.TRAIN_BATCH_SIZE = 32
_C.SOLVER.TEST_BATCH_SIZE = 32
_C.SOLVER.GRAD_ACC_STEPS = 2
_C.SOLVER.GRAD_CKPT = True
_C.SOLVER.FP16 = True
_C.SOLVER.EPOCHS = 10
_C.SOLVER.WEIGHT_DECAY = 1e-2


def get_cfg_defaults():
    return _C.clone()