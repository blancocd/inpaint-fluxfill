import os
import json
import shutil

input_dir = '/mnt/lustre/work/ponsmoll/pba534/Datasets/scans_inpaint_testset/3dcustom_4/'
output_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/data/'
f = open('/mnt/lustre/work/ponsmoll/pba534/inpaint/data/captions/scan_testset.json')
scans_dict = json.load(f)

for i, scan_dict in enumerate(scans_dict):
    scan_name = scan_dict['scan']
    dataset = scan_dict['dataset']

    scan_dir = os.path.join(input_dir, scan_name)
    flbr_img_path_src = os.path.join(scan_dir, 'mask_vis', 'flbr.png')
    flbr_img_path_dst = os.path.join(output_dir, 'seg', 'flbr', str(i) + '.png')
    shutil.copy(flbr_img_path_src, flbr_img_path_dst)
    
