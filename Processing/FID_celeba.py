import os
import numpy as np
import torch
import torchmetrics
from torchmetrics import FID
from glob import glob
import matplotlib.pyplot as plt

print(os.getcwd())

original_root = '../../CelebAHQ256_cleaned'
inpainted_root = '../../CelebAHQ256_inpainted_gatedconv'

# original_root = '../../celeba_256'
# inpainted_root = '../../celeba_256_inpainted_gatedconv'

original_fs = sorted(glob(original_root + '/*'))
inpainted_fs = sorted(glob(inpainted_root + '/*'))

print(len(original_fs))
print(len(inpainted_fs))

original_arr = np.asarray([plt.imread(f) for f in original_fs[:1000]])

print(original_arr.shape)

fid = FID(feature=64)
