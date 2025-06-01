import os
from tqdm import tqdm
from pathlib import Path

from cloth_masker import vis_mask
from diffusers.utils import load_image

local_dir = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill'
cluster_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint'
mydir = cluster_dir

for view in ['front', 'fb']:
    images_dir = f'{mydir}/data/images/{view}/'
    results_dir = f'{mydir}/data/upper_masks/{view}/'
    os.makedirs(results_dir, exist_ok=True)

    results_dir2 = f'{mydir}/data/applied_upper_masks_new/{view}/'
    os.makedirs(results_dir2, exist_ok=True)
    
    for image_fn in tqdm(os.listdir(images_dir)):
        person_image = load_image(os.path.join(images_dir, image_fn))
        mask = load_image(os.path.join(results_dir, f'{str(Path(image_fn).stem)}_upper.png'))
        mask = mask.convert('L')
        masked_person = vis_mask(person_image, mask)
        masked_person.save(os.path.join(results_dir2, f'{str(Path(image_fn).stem)}_upper.png'))
