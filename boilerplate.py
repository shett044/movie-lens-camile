# import torch
import utils as ut
import logging
from datetime import datetime
from pathlib import Path
import os

osjoin = os.path.join
save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

DIR_RESULTS = Path('results')
DIR_RESULTS.mkdir(exist_ok=True)
DIR_SAVE = DIR_RESULTS.joinpath(save)
DIR_SAVE.mkdir(parents=True, exist_ok=True)
DIR_PRETRAIN = DIR_SAVE.joinpath('pretrain')
DIR_PRETRAIN.mkdir(parents=True, exist_ok=True)
DIR_MODEL = DIR_SAVE.joinpath('model')
DIR_MODEL.mkdir(parents=True, exist_ok=True)

DIR_DATA = Path('data')
DIR_DATA_TMDB = Path('data').joinpath('tmdb')


DIR_RESULTS.joinpath("best_model").mkdir(exist_ok=True)
ut.setup_logging(str(DIR_SAVE.joinpath('log.txt')))
logInfo = logging.info
logDebug = logging.debug

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
# logInfo(f"{DEVICE = }")
# if torch.cuda.is_available():
#     print('Current cuda device:', torch.cuda.current_device())
#     print('Count of using GPUs:', torch.cuda.device_count())

# logInfo(f"{os.getcwd()=}")