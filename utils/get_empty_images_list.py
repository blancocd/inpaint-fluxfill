import os
import json
from PIL import Image

def get_non_transparent_images(json_path, input_dir):
    """
    Reads a JSON file containing a list of dicts with 'scan' keys.
    For each scan, checks all images in images/ subdir.
    Returns a list of image paths whose first pixel is not transparent.
    """
    non_transparent_images = []
    with open(json_path, 'r') as f:
        garments = json.load(f)
    for entry in garments:
        scan_name = entry['scan']
        images_dir = os.path.join(input_dir, scan_name, 'images')
        if not os.path.isdir(images_dir):
            continue

        img_path = os.path.join(images_dir, os.listdir(images_dir)[0])
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGBA')
                first_pixel = img.getpixel((0, 0))
                if len(first_pixel) == 4 and first_pixel[3] != 0:
                    non_transparent_images.append(scan_name)
        except Exception:
            continue
    return non_transparent_images

# Example usage:
json_path = '/mnt/lustre/work/ponsmoll/pba534/inpaint/data/captions/scan_testset.json'
input_dir = '/mnt/lustre/work/ponsmoll/pba534/Datasets/scans_inpaint_testset/3dcustom_4/'
result = get_non_transparent_images(json_path, input_dir)
with open('non_transparent_images.json', 'w') as f:
    json.dump(result, f, indent=2)
print(result)
