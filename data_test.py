import numpy as np

from torch.utils.data import DataLoader

from data import PCDataset

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dataset = PCDataset('miniv', image_size = 352, negative_prob = 0.2)
loader = DataLoader(dataset, 
                    batch_size = 1, 
                    num_workers = 0, 
                    shuffle = False, 
                    drop_last = False)


for x,y in loader:
  print(x[0].shape, x[1])
  break