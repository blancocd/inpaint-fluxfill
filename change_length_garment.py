import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image, make_image_grid
import json
import os.path as osp
from huggingface_hub import login
from utils.add_bottom_text import add_text_below_image_wrapped

import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
token = os.getenv("HUGGINGFACE_TOKEN")

login(token=token)
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# load base and mask image
base_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/test_caption_models'
results_dir = osp.join(base_dir, 'results', 'full_caption')

# Load scans from captions.json
json_fn = 'captions.json'
ext = 'png'
with open(osp.join(base_dir, json_fn), 'r') as f:
    scans = json.load(f)

for scan in scans:
    print(f"Testing scan {scan['scan_id']}")
    scan_id = str(scan['scan_id'])
    #if scan_id != 1:
    #    continue
    variants = scan["variants"]
    image = load_image(osp.join(base_dir, 'images', scan_id + '.' + ext))

    # First variant is the already existing one
    longer_sleeves_orig = variants[0]['longer_sleeves']
    longer_pants_orig = variants[0]['longer_pants']

    # changing sleeves
    prompt = "Full frontal view of " + variants[1]['caption']
    mask_upper = load_image(osp.join(base_dir, 'masks', scan_id + '_upper.' + ext))
    mask_lower = load_image(osp.join(base_dir, 'masks', scan_id + '_lower.' + ext))

    if 'real' not in json_fn:
        image = image.resize((1024,1024), Image.Resampling.LANCZOS)
        mask_upper = mask_upper.resize((1024,1024), Image.Resampling.LANCZOS)
        mask_lower = mask_lower.resize((1024,1024), Image.Resampling.LANCZOS)

    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_upper,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]
    # Create the image grid
    final_img = make_image_grid([image, mask_upper, gen_image], 1, 3)
    final_img = add_text_below_image_wrapped(final_img, prompt)
    sleeve_suffix = 'shorter_sleeves' if longer_sleeves_orig else 'longer_sleeves'
    final_img_fn = osp.join(results_dir, f"{scan_id}_{sleeve_suffix}.png")
    final_img.save(final_img_fn)

    # changing pants
    prompt = "Full frontal view of " + variants[2]['caption']
    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_lower,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    final_img = make_image_grid([image, mask_lower, gen_image], 1, 3)
    final_img = add_text_below_image_wrapped(final_img, prompt)
    pants_suffix = 'shorter_pants' if longer_pants_orig else 'longer_pants'
    final_img_fn = osp.join(results_dir, f"{scan_id}_{pants_suffix}.png")
    final_img.save(final_img_fn)



results_dir = osp.join(base_dir, 'results', 'garment_caption')

# Load scans from captions.json
with open(osp.join(base_dir, f'garment_{json_fn}'), 'r') as f:
    scans = json.load(f)

for scan in scans:
    print(f"Testing scan {scan['scan_id']}")
    scan_id = str(scan['scan_id'])
    variants = scan["variants"]
    image = load_image(osp.join(base_dir, 'images', scan_id + '.' + ext))
    if 'real' not in json_fn:
        image = image.resize((1024,1024), Image.Resampling.LANCZOS)

    # First variant is the already existing one
    longer_sleeves_orig = variants[0]['longer_sleeves']
    longer_pants_orig = variants[0]['longer_pants']

    # changing sleeves
    prompt = variants[1]['caption']
    mask_upper = load_image(osp.join(base_dir, 'masks', scan_id + '_upper.' + ext))
    mask_lower = load_image(osp.join(base_dir, 'masks', scan_id + '_lower.' + ext))

    if 'real' not in json_fn:
        image = image.resize((1024,1024), Image.Resampling.LANCZOS)
        mask_upper = mask_upper.resize((1024,1024), Image.Resampling.LANCZOS)
        mask_lower = mask_lower.resize((1024,1024), Image.Resampling.LANCZOS)

    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_upper,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]
    # Create the image grid
    final_img = make_image_grid([image, mask_upper, gen_image], 1, 3)
    final_img = add_text_below_image_wrapped(final_img, prompt)
    sleeve_suffix = 'shorter_sleeves' if longer_sleeves_orig else 'longer_sleeves'
    final_img_fn = osp.join(results_dir, f"{scan_id}_{sleeve_suffix}.png")
    final_img.save(final_img_fn)

    # changing pants
    prompt = variants[2]['caption']
    gen_image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_lower,
        height=image.height,
        width=image.width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    final_img = make_image_grid([image, mask_lower, gen_image], 1, 3)
    final_img = add_text_below_image_wrapped(final_img, prompt)
    pants_suffix = 'shorter_pants' if longer_pants_orig else 'longer_pants'
    final_img_fn = osp.join(results_dir, f"{scan_id}_{pants_suffix}.png")
    final_img.save(final_img_fn)
