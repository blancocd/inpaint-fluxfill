import os
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np

def proc_mask(mask):
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return mask

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = proc_mask(mask)
    print(np.max(mask), np.min(mask), np.sum(mask>0.5))
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def load_image(image) -> Image:
    image = Image.open(image)
    return image

local_dir = '/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill'
cluster_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint'
mydir = local_dir

for view in ['front', 'fb']:
    images_dir = f'{mydir}/data/images/{view}/'
    results_dir = f'{mydir}/data/upper_masks/{view}/'
    os.makedirs(results_dir, exist_ok=True)

    results_dir2 = f'{mydir}/data/applied_upper_masks_new/{view}/'
    os.makedirs(results_dir2, exist_ok=True)
    
    for image_fn in tqdm(os.listdir(images_dir)):
        person_image = load_image(os.path.join(images_dir, image_fn)).convert('RGB')
        mask = load_image(os.path.join(results_dir, f'{str(Path(image_fn).stem)}_upper.png')).convert('L')
        masked_person = vis_mask(person_image, mask)
        masked_person.save(os.path.join(results_dir2, f'{str(Path(image_fn).stem)}_upper.png'))
