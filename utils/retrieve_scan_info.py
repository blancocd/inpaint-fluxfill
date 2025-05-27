import os
import json

def filter_dictionary_by_images(input_dir, input_json_path, output_json_path):
    # Load the dictionary from the JSON file
    with open(input_json_path, 'r') as f:
        d = json.load(f)

    # Get the set of image names (without extensions) in the input directory
    image_names = sorted(
        {os.path.splitext(filename)[0] for filename in os.listdir(input_dir) if filename.endswith('.png')},
        key=lambda x: int(x) if x.isdigit() else float('inf')
    )

    # Create a subset of the dictionary with keys matching the image names
    filtered_dict = {}
    for key, value in d.items():
        if key in image_names:
            # Remove a specific key from the inner dictionary
            value.pop('caption', None)
            # Add a new key "inner" with an empty string
            value['inner'] = ""
            filtered_dict[key] = value

    # Save the filtered dictionary to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(filtered_dict, f, indent=4)

# Example usage
input_dir = '/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/data_2/images/fb'
input_json_path = '/home/cesar/Documents/WS2425/Avatar25/data/front_out_in_garments/metadata_train.json'
output_json_path = './out.json'
# filter_dictionary_by_images(input_dir, input_json_path, output_json_path)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def convert_dict_to_list(input_json_path, output_json_path):
    # Load the dictionary from the JSON file
    with open(input_json_path, 'r') as f:
        d = json.load(f)

    # Convert the dictionary to a sorted list of dictionaries
    sorted_list = [
        {**value} for i, (key, value) in enumerate(sorted(d.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')))
    ]

    # Save the list of dictionaries to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(sorted_list, f, indent=4)

# Example usage
# convert_dict_to_list('/home/cesar/Documents/WS2425/Avatar25/inpaint-fluxfill/utils/out.json', './out_list.json')