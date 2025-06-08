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
    outer_garm_type = outer_garm.split(' ')[-1].replace('-', ' ')
    outer_garm_types = ['jacket', 'cardigan', 'sweater', 'coat']
    outer_garm_types = [outer_garm_type] + [ogt for ogt in outer_garm_types if ogt not in outer_garm_type]

    if len(outer_garm_types) > 1:
        outer_garm_types_str = ', '.join(outer_garm_types[:-1]) + f', or {outer_garm_types[-1]}'
    else:
        outer_garm_types_str = outer_garm_types[0]

    
    prompt_short_outer_desc = f'Answer yes if the person in the image is wearing a {outer_garm_type}. Answer no if the person is only wearing a {inner_garm}, shirt, t-shirt, or blouse.'
    prompt_long_outer_desc = f'Answer yes if the person in the image is wearing a {outer_garm_types_str}. Answer no if the person is only wearing a {inner_garm}, shirt, t-shirt, or blouse.'
    answers = []
    for prompt in [prompt_short_outer_desc, prompt_long_outer_desc]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description for top body garments."}]
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
        answers.append(decoded)
    
    result_dict = {
        'short' : {
            'prompt': prompt_short_outer_desc,
            'full_answer' : answers[0],
            'successful': 'no' in answers[0].lower()
        },
        'long' : {
            'prompt': prompt_long_outer_desc,
            'full_answer' : answers[1],
            'successful': 'no' in answers[1].lower()
        }
    }

    # The short prompt is more reliable. When the answers to the short and long prompts disagree, it is worth seeing 
    # why is this, still the long prompt is too strict and is wrong around 7/10 of the time while the short prompt 3/10
    # So we use this same weights to acount for disagreements.

    count += 0.7*int('no' in answers[0].lower()) + 0.3*int('no' in answers[1].lower())
    results_list.append(result_dict)

print(count)

with open('results_nonmasked_garment.json', 'w') as f:
    json.dump(results_list, f, indent=2)