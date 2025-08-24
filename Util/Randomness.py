import os
import random
import numpy as np
import torch

import Config

os.environ['PYTHONHASHSEED'] = str(Config.seed)

random.seed(Config.seed)
np.random.seed(Config.seed)
rng = np.random.default_rng(Config.seed)

torch.manual_seed(Config.seed)
torch.cuda.manual_seed(Config.seed)
torch.cuda.manual_seed_all(Config.seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    random.seed(Config.seed)
    np.random.seed(Config.seed)

generator = torch.Generator()
generator.manual_seed(Config.seed)
