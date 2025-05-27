import torch
from pipelines.flux_fill_two_masks import FluxFillPipeline
from diffusers.utils import load_image, make_image_grid
import json
import os.path as osp
from huggingface_hub import login

import numpy as np
import os
from utils.add_bottom_text import add_text_below_image_wrapped

token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# load base and mask image
base_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/'
data_dir = os.path.join(base_dir, 'data')
view = 'fb'
task = 'remove'
mask_type = 'both_masks'
results_dir = osp.join(base_dir, 'results', view, task, mask_type)
os.makedirs(results_dir, exist_ok=True)

# Load scans from captions.json
json_fn = 'captions_remove_fb.json'
with open(osp.join(base_dir, json_fn), 'r') as f:
    scans = json.load(f)

specific_scans = ['1828', '1868']
for mask_interpolation_param in [0.5]: #np.linspace(0.0, 1.0, 5):
    results_dir = osp.join(base_dir, 'results', view, task, mask_type, 'miprm'+str(mask_interpolation_param))
    os.makedirs(results_dir, exist_ok=True)
    for scan in scans:
        print(f"Testing scan {scan['scan_id']}")
        scan_id = str(scan['scan_id'])
        if specific_scans is not None and scan_id in specific_scans:
           continue 
        prompts = scan["prompts"]
        image = load_image(osp.join(base_dir, 'images', view, scan_id + '.png'))

        prompt = prompts[0]
        general_mask_upper = load_image(osp.join(base_dir, 'masks', view, scan_id + '_upper.png'))
        specific_mask_upper = load_image(osp.join(base_dir, 'masks', view, scan_id + '_upper_garment.png'))
        
        gen_image = pipe(
            prompt=prompt,
            image=image,
            mask_image_general=general_mask_upper,
            mask_image_specific=specific_mask_upper,
            height=image.height,
            width=image.width,
            guidance_scale=30,
            mask_interpolation_param=mask_interpolation_param,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        final_img = make_image_grid([image, general_mask_upper, specific_mask_upper, gen_image], 1, 4)
        final_img = add_text_below_image_wrapped(final_img, prompt)
        final_img_fn = osp.join(results_dir, f"{scan_id}_undergarment.png")
        final_img.save(final_img_fn)
        
        prompt = ''
        gen_image = pipe(
            prompt=prompt,
            image=image,
            mask_image_general=general_mask_upper,
            mask_image_specific=specific_mask_upper,
            height=image.height,
            width=image.width,
            guidance_scale=30,
            mask_interpolation_param=mask_interpolation_param,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        final_img = make_image_grid([image, general_mask_upper, specific_mask_upper, gen_image], 1, 4)
        final_img = add_text_below_image_wrapped(final_img, prompt)
        final_img_fn = osp.join(results_dir, f"{scan_id}_noprompt.png")
        final_img.save(final_img_fn)
