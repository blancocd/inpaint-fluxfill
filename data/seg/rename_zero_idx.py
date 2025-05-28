import os

# Directory containing the images
input_dir = '/mnt/lustre/work/ponsmoll/pba534/inpaint/data/seg/'

for typ in ['fb', 'front']:
    # Get a list of all files in the directory
    directory = os.path.join(input_dir, typ)
    files = sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else float('inf'))

    # Rename files from 1 to 100 to 0 to 99
    for file in files:
        # Extract the file name and extension
        name, ext = os.path.splitext(file)
        
        # Check if the name is a digit
        if name.isdigit():
            old_index = int(name)
            new_index = old_index - 1
            
            # Generate the new file name
            new_name = f"{new_index}{ext}"
            
            # Rename the file
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)

    print("Renaming completed.")