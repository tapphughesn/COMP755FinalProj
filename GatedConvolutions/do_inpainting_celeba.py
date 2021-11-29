import os
from glob import glob
import warnings

cmd = 'cd ../../generative_inpainting; python test.py --image ../celeba_256_masked_png/{} --mask ../celeba_256_masks_pngs_for_gatedconv/{} --output ../celeba_256_inpainted_gatedconv/{} --checkpoint model_logs/release_celeba_hq_256_deepfill_v2 >/dev/null 2>&1'

img_filenames = sorted(glob('../../celeba_256_masked_png/*'))
mask_filenames = sorted(glob('../../celeba_256_masks_pngs_for_gatedconv/*'))

save_dir = '../../celeba_256_inpainted_gatedconv/'

assert(len(img_filenames) == len(mask_filenames))

# inefficient for running inference on multiple images but I didn't want to dive into the code.
# ideally you would compile the tensorflow model graph one time and run on multiple images

for i in range(len(img_filenames)):
    img_f = img_filenames[i]
    mask_f = mask_filenames[i]
    savename = os.path.join(save_dir, os.path.basename(img_f) + '_inpainted.png')
    if os.path.isfile(savename):
        continue
    # print(cmd.format(os.path.basename(img_f), os.path.basename(mask_f), os.path.basename(savename)))
    os.system(cmd.format(os.path.basename(img_f), os.path.basename(mask_f), os.path.basename(savename)))
