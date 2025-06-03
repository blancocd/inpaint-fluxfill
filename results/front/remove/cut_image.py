from PIL import Image
import os

input_dirs = ['/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/results/front/remove/garment_mask', 
              '/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/results/front/remove/upper_mask']

for input_dir in input_dirs:
    for input_path in os.listdir(input_dir):
        image_fn = os.path.join(input_dir, input_path)

        img = Image.open(image_fn)
        width, height = img.size

        # Calculate new width (one third of original)
        new_width = width // 3

        # Define the crop box for the last third (left, upper, right, lower)
        crop_box = (width - new_width, 0, width, height)

        # Crop the image
        cropped_img = img.crop(crop_box)

        # Save the cropped image
        cropped_img.save(image_fn)
