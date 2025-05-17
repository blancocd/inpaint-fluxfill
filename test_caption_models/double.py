import os
from PIL import Image

# Path to the folder containing the images
folder_path = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/test_caption_models/images/fb'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        file_path = os.path.join(folder_path, filename)
        
        # Open the image
        with Image.open(file_path) as img:
            # Double the size of the image
            new_size = (img.width * 2, img.height * 2)
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save the resized image, overwriting the original
            resized_img.save(file_path)

print("All images have been resized.")