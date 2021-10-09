import os
from glob import glob

image_filenames = glob('../../CelebAHQ256/*')

count = 0

for filepath in image_filenames:
    mask_dir = '../../CelebAHQ256_masks/'
    mask_path = os.path.join(mask_dir, os.path.basename(filepath) + '_mask.npy')
    if not os.path.isfile(mask_path):
        count += 1
        print('found img without mask', count)
        os.remove(filepath)
