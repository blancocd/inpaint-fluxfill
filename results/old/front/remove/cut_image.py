from PIL import Image
import os

input_dirs = ['/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/results/front/remove/garment_mask', 
              '/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/results/front/remove/upper_mask']

for input_dir in input_dirs:
    for input_path in os.listdir(input_dir):
        image_fn = os.path.join(input_dir, input_path)
        if os.path.isdir(image_fn):
            continue
        os.makedirs(os.path.join(input_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(input_dir, 'results'), exist_ok=True)

        img = Image.open(image_fn)
        width, height = img.size
        new_width = width // 3

        original_crop_box = (0, 0, new_width, height)
        cropped_img = img.crop(original_crop_box)
        number_part = os.path.splitext(os.path.basename(image_fn))[0].split('_')[0]
        cropped_img.save(os.path.join(input_dir, 'original', number_part + ".png"))

        result_crop_box = (width - new_width, 0, width, height)
        cropped_img = img.crop(result_crop_box)
        number_part = os.path.splitext(os.path.basename(image_fn))[0].split('_')[0]
        cropped_img.save(os.path.join(input_dir, 'results', number_part + ".png"))

