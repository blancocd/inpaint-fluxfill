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

images_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/data/images/front'
results_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/results/nonmasked_garment/front'

with open('/mnt/lustre/work/ponsmoll/pba534/inpaint/data/captions/scan_testset.json', 'r') as f:
    garments = json.load(f)

results_list = []
count = 0
for idx, garment_dict in tqdm(enumerate(garments)):
    scan_id = str(idx)
    inner_garm = garment_dict['inner']
    outer_garm = garment_dict['outer']
    prompt = f'Is the person in the image wearing a jacket, cardigan, sweater, coat or a {outer_garm}? Answer yes if that is the case and no if the person is only wearing a shirt, t-shirt, blouse or a {inner_garm}.'
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": os.path.join(results_dir, f'{scan_id}_prompt.png')},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    
    result_dict = {
        'full_answer' : decoded,
        'successful': 'no' in decoded[0].lower()
    }
    count += int('no' in decoded[0].lower())
    results_list.append(result_dict)

print(count)

with open('results_nonmasked_garment.json', 'w') as f:
    json.dump(results_list, f, indent=2)