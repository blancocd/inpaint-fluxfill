from transformers import pipeline, AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "image-text-to-text",
#     model="google/gemma-3-4b-it",
#     device="cuda",
#     torch_dtype=torch.bfloat16
# )

input_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/results/nonmasked_garment/front'
mask_types = ['garment_mask', 'upper_mask']


json_path = os.path.join('/mnt/lustre/work/ponsmoll/pba534/inpaint/data/captions/remove_front.json')
with open(json_path, "r") as f:
    garments = json.load(f)

def get_key(garments, scan_id, key):
    for item in garments:
        if item['scan_id'] == scan_id:
            return item[key]

for mask_type in mask_types:
    orig_dir = os.path.join(input_dir, mask_type, 'original')
    resu_dir = os.path.join(input_dir, mask_type, 'results')
    for image_fn in tqdm(os.listdir(orig_dir)):
        scan_id = os.path.splitext(os.path.basename(image_fn))[0].split('.')[0]

        # orig_image = Image.open(os.path.join(orig_dir, image_fn)).convert('RGB')
        # resu_image = Image.open(os.path.join(resu_dir, image_fn)).convert('RGB')

        inner_garment = get_key(garments, scan_id, 'inner')
        outer_garment = get_key(garments, scan_id, 'outer')
        gen_outer_garment = get_key(garments, scan_id, 'general_outer')

        prompt_two_images = f'In the first image there is a person with {inner_garment} as well as a {outer_garment} on top. In the second image the outer garment should have been removed and there should be no outer garment at all. Answer with a simple yes if the person is still wearing a {gen_outer_garment} in the second image. Answer with a simple no if the person is only wearing {inner_garment} in the second image.'
        prompt_one_image = 'Is the person in the image wearing a jacket, cardigan, sweater, or coat? Answer yes if that is the case and no if the person is only wearing a shirt, t-shirt or blouse.'
        messages = [[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": os.path.join(orig_dir, image_fn)},
                    {"type": "image", "path": os.path.join(resu_dir, image_fn)},
                    {"type": "text", "text": prompt_two_images}
                ]
            }
        ],
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": os.path.join(resu_dir, image_fn)},
                    {"type": "text", "text": prompt_one_image}
                ]
            }
        ]
        ]

        # outputs = pipe(text=messages, images=[orig_image, resu_image], max_new_tokens=200)
        # print(outputs[0]["generated_text"])

        outputs = []
        for message in messages:
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            outputs.append(decoded)
        print(scan_id, mask_type, f'long prompt: {outputs[0]}', f'short prompt: {outputs[1]}')

        
