import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image, make_image_grid
import json
import os.path as osp
from huggingface_hub import login

import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
token = os.getenv("HUGGINGFACE_TOKEN")

# next is to only describe the inpainted region


def add_text_below_image_wrapped(
    img, text, font_size=30, fill=(0,0,0),
    padding=10, bg_color=(255,255,255), max_width=None, line_spacing=4
):
    # load font
    font = ImageFont.load_default(size=font_size)

    # determine max text width
    if max_width is None:
        max_width = img.width - 2 * padding

    # prepare a drawing context on a dummy image to measure text
    dummy_draw = ImageDraw.Draw(img)

    # wrap text into lines fitting max_width
    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=1000)  # initial break: no limit
        # refine by measuring
        refined = []
        for line in wrapped:
            words = line.split(" ")
            current = ""
            for word in words:
                test = current + (" " if current else "") + word
                left, top, right, bottom = dummy_draw.textbbox((0,0), test, font=font)
                if right - left <= max_width:
                    current = test
                else:
                    if current:
                        refined.append(current)
                    current = word
            if current:
                refined.append(current)
        lines.extend(refined)

    # measure total text block height
    heights = []
    widths = []
    for line in lines:
        l, t, r, b = dummy_draw.textbbox((0,0), line, font=font)
        widths.append(r - l)
        heights.append(b - t)
    text_block_height = sum(heights) + line_spacing * (len(lines)-1)
    text_block_width = max(widths) if widths else 0

    # new canvas size
    new_width = max(img.width, text_block_width + 2*padding)
    new_height = img.height + padding + text_block_height + padding

    # create new image
    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    # paste original centered horizontally
    x_offset = (new_width - img.width) // 2
    new_img.paste(img, (x_offset, 0))

    # draw each line centered below
    draw = ImageDraw.Draw(new_img)
    y = img.height + padding
    for i, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0,0), line, font=font)
        line_width = right - left
        x = (new_width - line_width) / 2 - left
        draw.text((x, y - top), line, font=font, fill=fill)
        y += (bottom - top) + line_spacing

    return new_img


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
