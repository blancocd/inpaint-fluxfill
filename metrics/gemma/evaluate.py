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

results_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/results/nonmasked_garment/front'

results_list = []
count_noprompt = 0
count_prompt = 0
for idx in tqdm(range(100)):
    scan_id = str(idx)

    prompt = 'Is the person in the image wearing a jacket, cardigan, sweater, or coat? Answer yes if that is the case and no if the person is only wearing a shirt, t-shirt or blouse.'
    messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": os.path.join(results_dir, f'{scan_id}_noprompt.png')},
                {"type": "text", "text": prompt}
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
                {"type": "image", "path": os.path.join(results_dir, f'{scan_id}_prompt.png')},
                {"type": "text", "text": prompt}
            ]
        }
    ] 
    ]

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
    result_dict = {
        'full_answer' : {
            'prompt' : outputs[0],
            'no_prompt' : outputs[1]
        },
        'succesful': {
            'prompt' : 'no' in outputs[0].lower(),
            'no_prompt' : 'no' in outputs[1].lower(),
        }
    }
    count_prompt += int('no' in outputs[0].lower())
    count_noprompt += int('no' in outputs[1].lower())
    results_list.append(result_dict)

print(count_noprompt, count_prompt)

with open('results_nonmasked_garment.json', 'w') as f:
    json.dump(results_list, f, indent=2)