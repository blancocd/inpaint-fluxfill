from PIL import Image
import os
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import Path

catvton_repo = "zhengchong/CatVTON"
repo_path = snapshot_download(repo_id=catvton_repo, allow_patterns=['*DensePose*', '*SCHP*'])
print(repo_path)

from cloth_masker import AutoMasker, vis_mask
from diffusers.utils import load_image

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

local_dir = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill'
cluster_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint'
mydir = cluster_dir

for view in ['front', 'fb']:
    images_dir = f'{mydir}/data/images/{view}/'
    results_dir = f'{mydir}/data/outer_masks/{view}/'
    os.makedirs(results_dir, exist_ok=True)

    results_dir2 = f'{mydir}/data/applied_outer_masks/{view}/'
    os.makedirs(results_dir2, exist_ok=True)

    # mask_types = ['upper', 'outer']
    mask_types = ['outer']
    for image_fn in tqdm(os.listdir(images_dir)):
        person_image = load_image(os.path.join(images_dir, image_fn))
        # new_width = (person_image.width // 4) * 4
        # new_height = (person_image.height // 4) * 4
        # person_image = person_image.resize((new_width, new_height))
        for mask_type in mask_types:
            return_dir = automasker(person_image, mask_type, upper=True)
            mask = return_dir['mask']
            densepose = return_dir['densepose']
            schp_lip = return_dir['schp_lip']
            schp_atr = return_dir['schp_atr']

            mask.save(os.path.join(results_dir, f'{str(Path(image_fn).stem)}_{mask_type}.png'))
   
            masked_person = vis_mask(person_image, mask, partial_transparency=True)
            masked_person.save(os.path.join(results_dir2, f'{str(Path(image_fn).stem)}_{mask_type}.png'))
