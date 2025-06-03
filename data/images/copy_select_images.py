import os
import shutil

# Set your source and destination directories
src_dir = '/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/data/images/fb'
dst_dir = '/home/blancocd/Development/SU25/Avatar25/inpaint-fluxfill/data/images/fb_refine_inner_mask'

# List of integers for the images you want to copy
image_numbers = [17, 18, 19, 23, 24, 34, 37, 38, 39, 40, 43, 48, 51, 53, 54, 58, 66, 69, 74, 75, 77, 83, 88, 89, 93, 97, 98, 99, 26, 28, 46, 56, 76]  # Example list

os.makedirs(dst_dir, exist_ok=True)

for num in image_numbers:
    filename = f"{num}.png"
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"File not found: {src_path}")