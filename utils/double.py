import os
from PIL import Image

# Path to the folder containing the images
folder_path = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/data/'

# Walk through all directories and files in the folder
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Check if the image size is 512x512
                if img.size == (512, 512):
                    # Double the size of the image
                    new_size = (1024, 1024)
                    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save the resized image, overwriting the original
                    resized_img.save(file_path)

print("All applicable images have been resized.")


