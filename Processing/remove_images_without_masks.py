import os
from glob import glob

image_filenames = glob('../../img_align_celeba/*')

count = 0

for filepath in image_filenames:
    mask_dir = '../../img_align_celeba_masks/'
    mask_path = os.path.join(mask_dir, os.path.basename(filepath) + '_mask.npy')
    if not os.path.isfile(mask_path):
        count += 1
        print('found img without mask', count)
        os.remove(filepath)
