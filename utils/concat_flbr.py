import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import json

def get_human_height_width(img: Image, background = 1):
    img = img.convert("L")
    img_arr = np.array(img)

    img = img_arr > background 
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def crop_img(img: Image, dims: tuple | None = None) -> np.array:
    img = img.convert("RGB")
    if dims is None:
        ceiling, floor, left, right = get_human_height_width(img)
    else:
        ceiling, floor, left, right = dims

    cropped_img = np.array(img)[ceiling:floor, left:right]

    if dims is None:
        dims = ceiling, floor, left, right
        return cropped_img, dims
    else:
        return cropped_img


# background is 0 for mask vis images and depth, 255 for images
def concat_imgs_width(imgs: list[np.ndarray], pixel_sep=20, background = 0, shape: tuple = (1024, 1024)):
    channels, widths = [], []
    max_height = 0
    for img in imgs:
        height, width, c = img.shape
        widths.append(width)
        channels.append(c)
        max_height = height if height > max_height else max_height
    added_width = sum(widths) + (len(imgs) + 1) * pixel_sep
    max_height += 2 * pixel_sep
    assert all(i == channels[0] for i in channels)

    newsize = max(added_width, max_height)
    pixel_sep = int((newsize - sum(widths)) / (len(imgs)+1))
    concat_imgs_arr = np.ones((newsize, newsize, channels[0])) * background
    for i, img in enumerate(imgs):
        start_height = int(newsize/2-img.shape[0]/2)
        start_width = (i+1)*pixel_sep + sum(widths[:i])
        end_height = start_height + img.shape[0]
        end_width = start_width + img.shape[1]
        concat_imgs_arr[start_height: end_height, start_width: end_width] = img
    
    concat_imgs_arr = concat_imgs_arr.astype(np.uint8)
    return Image.fromarray(concat_imgs_arr).resize(shape, resample=Image.Resampling.LANCZOS)


def bgr_transp_to_white(image, bgr):
    if image.mode != 'RGBA':
        return image
    if bgr:
        img_rgb = image.split()[:3][::-1]  # Reverse RGB channels
        img_a = image.split()[3]  # Extract alpha channel
        image = Image.merge("RGBA", (*img_rgb, img_a))  # Merge back to RGBA

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image = new_image.convert("RGB")
    return new_image

    
def concat_imgs(scan_dir, bgr, fb_id):
    imgs_dir = os.path.join(scan_dir, 'images')
    segm_dir = os.path.join(scan_dir, 'mask_vis')

    # The cropping is determined by the segmentation images as they have clear limits    
    imgs, segms = [], []
    for id in fb_id:
        segm = Image.open(os.path.join(segm_dir, f"images_000{str(id)}.png"))
        cropped_segm, dims = crop_img(segm)
        img = Image.open(os.path.join(imgs_dir, f"images_000{str(id)}.png"))
        img = bgr_transp_to_white(img, bgr)
        cropped_img = crop_img(img, dims)
        imgs.append(cropped_img)
        segms.append(cropped_segm)

    concat_imgs_width(imgs, background=255).save(os.path.join(imgs_dir, 'flb.png'))
    concat_imgs_width(segms).save(os.path.join(segm_dir, 'flb.png'))


def get_dataset(scans_data, scan_name):
    for dct in scans_data:
        if dct['scan'] == scan_name:
            return dct['dataset']

datasets = ['2k2k',     'axyz',     'closedi',  'renderpeople_0',   'renderpeople_1',   'thuman2',  'treddy',   'twindom_10k']
bgrs = [    False,      True,       False,      True,               True,               True,       True,       True]
fb_idx = [  [0,1,2,3],  [0,1,2,3],  [0,1,2,3],  [0,1,2,3],          [1,2,3,0],          [0,1,2,3],  [0,1,2,3],  [0,1,2,3]]
fb_idx = [  [0,1,2],  [0,1,2],  [0,1,2],  [0,1,2],          [1,2,3],          [0,1,2],  [0,1,2],  [0,1,2]]

input_dir = '/mnt/lustre/work/ponsmoll/pba534/Datasets/scans_inpaint_testset/3dcustom_4/'
f = open('/mnt/lustre/work/ponsmoll/pba534/inpaint/data/captions/scan_testset.json')
scans_dict = json.load(f)

for scan_dict in tqdm(scans_dict):
    scan_name = scan_dict['scan']
    dataset = scan_dict['dataset']

    bgr = bgrs[datasets.index(dataset)]
    fb_id = fb_idx[datasets.index(dataset)]

    scan_dir = os.path.join(input_dir, scan_name)
    concat_imgs(scan_dir, bgr, fb_id)
