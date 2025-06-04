import torch
from pipelines.flux_fill_two_masks import FluxFillPipeline
from diffusers.utils import load_image, make_image_grid
import json
import os.path as osp
from huggingface_hub import login

from tqdm import tqdm
import numpy as np
import os
from utils.add_bottom_text import add_text_below_image_wrapped
from PIL import Image

def nand_image(img1, img2):
    img1_np = np.array(img1.convert('L'))
    img2_np = np.array(img2.convert('L'))

    # Ensure masks are binary (0 or 255)
    img1_bin = (img1_np > 127).astype(np.uint8)
    img2_bin = (img2_np > 127).astype(np.uint8)

    # Compute NAND: invert (img1 AND img2)
    nand_mask = np.logical_xor(img1_bin, img2_bin).astype(np.uint8) * 255

    # Convert back to PIL Image
    return Image.fromarray(nand_mask, mode='L')

token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# load base and mask image
base_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/'
data_dir = os.path.join(base_dir, 'data')
view = 'front'

# Load scans from captions.json
json_fn = 'scan_testset.json'
with open(osp.join(data_dir, 'captions', json_fn), 'r') as f:
    scans = json.load(f)

specific_scans = None # ['1828', '1868']
results_dir = osp.join(base_dir, 'results', 'nonmasked_garment', view)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'grid'), exist_ok=True)

for scan_id, scan in tqdm(enumerate(scans)):
    scan_id = str(scan_id)
    if specific_scans is not None and scan_id in specific_scans:
        continue 
    
    image = load_image(osp.join(data_dir, 'images', view, scan_id + '.png'))

    upper_mask = load_image(osp.join(data_dir, 'masks', 'upper', view, scan_id + '.png'))
    inner_mask = load_image(osp.join(data_dir, 'masks', 'inner', view, scan_id + '.png'))
    outer_mask = nand_image(upper_mask, inner_mask)
    
    gen_image = pipe(
        prompt=scan['inner'],
        image=image,
        mask_image_general=upper_mask,
        mask_image_specific=outer_mask,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        mask_interpolation_param=0.,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    gen_image.save(osp.join(results_dir, f"{scan_id}_prompt.png"))
    img_grid = make_image_grid([image, upper_mask, outer_mask, gen_image], 1, 4)
    img_grid = add_text_below_image_wrapped(img_grid, scan['inner'])
    img_grid.save(osp.join(results_dir, 'grid', f"{scan_id}_prompt.png"))
    
    gen_image = pipe(
        prompt='',
        image=image,
        mask_image_general=upper_mask,
        mask_image_specific=outer_mask,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        mask_interpolation_param=0.,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    gen_image.save(osp.join(results_dir, f"{scan_id}_noprompt.png"))
    img_grid = make_image_grid([image, upper_mask, outer_mask, gen_image], 1, 4)
    final_img_fn = osp.join(results_dir, 'grid', f"{scan_id}_noprompt.png")
    img_grid.save(final_img_fn)
