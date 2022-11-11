import numpy as np

from torch.utils.data import DataLoader

from data import PCDataset

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dataset = PCDataset('train', image_size = 352, negative_prob = 0.2)
loader = DataLoader(dataset, 
                    batch_size = 32, 
                    num_workers = 1, 
                    shuffle = False, 
                    drop_last = False)