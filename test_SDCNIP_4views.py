import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image, make_image_grid
import os.path as osp
import os

# load ControlNet
myckpt = "/mnt/lustre/work/ponsmoll/pba534/controlnet_train/output_fb_seg/"
controlnet = ControlNetModel.from_pretrained(myckpt)

# pass ControlNet to the pipeline
pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float32)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
example_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/examples/4views'
results_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/results/'
for scan_id in ['116', '117', '1700', '1701', '1794', '1795']:
    os.makedirs(osp.join(results_dir, scan_id), exist_ok=True)
    if int(scan_id)%2==0:
        prompt_start = "Front view and back view"
    else:
        continue
        prompt_start = "Left view and right view"

    if scan_id in ['1794','1795']:
        gender = 'man'
    else:
        gender = 'woman'
    # tiny clothing and complex pattern
    init_image = load_image(osp.join(example_dir, scan_id, 'image.png'))
    mask_image = load_image(osp.join(example_dir, scan_id, 'mask1.png'))
    control_image = load_image(osp.join(example_dir, scan_id, 'segdep.png'))
    prompt = prompt_start + f" of the same {gender} wearing a marathon athletic shirt which has sponsors printed on it"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image, controlnet_conditioning_scale=1.2).images[0]
    final_img = make_image_grid([init_image, mask_image, control_image, image], 1, 4)
    final_img.save(osp.join(results_dir, scan_id, '1.png'))

    # sweater as new layer
    init_image = load_image(osp.join(example_dir, scan_id, 'image.png'))
    mask_image = load_image(osp.join(example_dir, scan_id, 'mask2.png'))
    control_image = load_image(osp.join(example_dir, scan_id, 'segdep.png'))
    prompt = prompt_start + f" of the same {gender} wearing a grey sweater"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image, controlnet_conditioning_scale=1.2).images[0]
    final_img = make_image_grid([init_image, mask_image, control_image, image], 1, 4)
    final_img.save(osp.join(results_dir, scan_id, '2.png'))

    # back to tiny clothing from sweater form before
    init_image = image
    mask_image = load_image(osp.join(example_dir, scan_id, 'mask2.png'))
    control_image = load_image(osp.join(example_dir, scan_id, 'segdep.png'))
    prompt = prompt_start + f" of the same {gender} wearing a sleeveless shirt"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image, controlnet_conditioning_scale=1.2).images[0]
    final_img = make_image_grid([init_image, mask_image, control_image, image], 1, 4)
    final_img.save(osp.join(results_dir, scan_id, '3.png'))

    # reconstruct shorts
    init_image = load_image(osp.join(example_dir, scan_id, 'image_no_shorts.png'))
    mask_image = load_image(osp.join(example_dir, scan_id, 'mask3.png'))
    control_image = load_image(osp.join(example_dir, scan_id, 'segdep.png'))
    color = "grey" if scan_id in ['1794', '1795'] else 'white'
    prompt = prompt_start + f" of the same {gender} wearing green shorts and a " + color + " top"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image, controlnet_conditioning_scale=1.2).images[0]
    final_img = make_image_grid([init_image, mask_image, control_image, image], 1, 4)
    final_img.save(osp.join(results_dir, scan_id, '4.png'))
    
    if int(scan_id) % 2 == 0:
        # blazer as new layer
        init_image = load_image(osp.join(example_dir, scan_id, 'image.png'))
        mask_image = load_image(osp.join(example_dir, scan_id, 'mask4.png'))
        control_image = load_image(osp.join(example_dir, scan_id, 'segdep.png'))
        prompt = prompt_start + f" of the same {gender} wearing a blue vneck blazer"
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, control_image=control_image, controlnet_conditioning_scale=1.2).images[0]
        final_img = make_image_grid([init_image, mask_image, control_image, image], 1, 4)
        final_img.save(osp.join(results_dir, scan_id, '5.png'))

