import os
import shutil

def copy_matching_files(input_dir, src_dir, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Get the list of PNG filenames in the input directory
    png_filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Move the files from src_dir to out_dir if they match the filenames in input_dir
    for filename in png_filenames:
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(out_dir, filename)
        print(src_file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)

    print("Selected files have been moved.")

view = 'front'
type = 'seg'
input_dir = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/data/images/front'
src_dir = f'/home/cesar/Documents/WS2425/Avatar25/data/{view}_out_in_garments/train/{type}'
out_dir = f'/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/data/{type}/{view}'
# copy_matching_files(input_dir, src_dir, out_dir)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def rename_files_sequentially(directory):
    # Get the list of files in the directory and sort them
    files = sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else float('inf'))
    print(len(files))
    
    # Rename each file sequentially
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{i}{os.path.splitext(filename)[1]}"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

# Example usage
for view in ['front', 'fb']:
    for type in ['seg', 'images']:
        input_dir = f'/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/data/{type}/{view}'
        rename_files_sequentially(input_dir)
