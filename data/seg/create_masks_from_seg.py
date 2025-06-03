import os
import numpy as np
from PIL import Image
from collections import Counter
import sys
from tqdm import tqdm

def get_unique_colors(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    pixels = arr.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    color_list = [tuple(color) for color in colors]
    return color_list, counts

def print_colors(colors, counts):
    print("Unique colors in the image:")
    for idx, (color, count) in enumerate(zip(colors, counts)):
        print(f"{idx}: {color} (count: {count})")

def create_mask(image_path, target_color, output_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    mask = np.all(arr == target_color, axis=-1).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Define target colors (convert to 0-255 int tuples)
    target_colors = [
    tuple(int(round(c * 255)) for c in [0.17254902, 0.62745098, 0.17254902]),
    tuple(int(round(c * 255)) for c in [0.83921569, 0.15294118, 0.15686275])
    ]
    tolerance = 10  # You can adjust this value for color closeness
    for fname in tqdm(image_files):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        for target in target_colors:
            close = np.all(np.abs(arr - target) <= tolerance, axis=-1)
            mask = np.logical_or(mask, close)
        mask = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(output_dir, fname))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_masks_from_seg.py <input_dir> <output_dir>")
    else:
        main(sys.argv[1], sys.argv[2])