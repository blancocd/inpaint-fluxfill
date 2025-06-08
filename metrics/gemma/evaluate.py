from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import json
import os
from tqdm import tqdm
from diffusers.utils import load_image, make_image_grid

model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

images_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/data/images/front'
results_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/results/nonmasked_garment/front'

results_list = []
count_noprompt = 0
count_prompt = 0
for idx in tqdm(range(100)):
    scan_id = str(idx)

    img = load_image(os.path.join(images_dir, scan_id + '.png'))
    img_prompt = load_image(os.path.join(results_dir, f'{scan_id}_prompt.png'))
    img_noprompt = load_image(os.path.join(results_dir, f'{scan_id}_noprompt.png'))
    
    os.makedirs(os.path.join(results_dir, 'grid2'), exist_ok=True)
    make_image_grid([img, img_prompt], 1, 2).save(os.path.join(results_dir, 'grid2', f'{scan_id}_prompt.png'))
    make_image_grid([img, img_noprompt], 1, 2).save(os.path.join(results_dir, 'grid2', f'{scan_id}_noprompt.png'))

    # prompt = 'Is the person in the image wearing a jacket, cardigan, sweater, or coat? Answer yes if that is the case and no if the person is only wearing a shirt, t-shirt or blouse.'
    prompt = 'In the image, the person on the left is wearing a jacket, cardigan, sweater, or coat. In the right, the same person **should** not be wearing a top garment and instead only be wearing a shirt, t-shirt or blouse. I am testing a model which removes top garments, is the person on the right wearing the top garment from the left image? Answer yes if that is the case and no if the person is wearing the inner garment.'
    messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant focused on clothing description."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": os.path.join(results_dir, 'grid2', f'{scan_id}_prompt.png')},
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
                {"type": "image", "path": os.path.join(results_dir, 'grid2', f'{scan_id}_noprompt.png')},
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