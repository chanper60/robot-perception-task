import os
import json
from PIL import Image
from tqdm import tqdm

def update_image_dimensions(coco_json_path, image_folder_path):
    # Load the COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Dictionary to map image IDs to their correct dimensions
    image_id_to_dimensions = {}
    wrong_dimensions_count = 0

    # Iterate over images in the COCO JSON
    for image_info in tqdm(coco_data.get('images', []), desc="Processing images"):
        image_filename = image_info['file_name']
        image_path = os.path.join(image_folder_path, image_filename)

        if os.path.exists(image_path):
            # Get the actual dimensions of the image
            with Image.open(image_path) as img:
                width, height = img.size

            # Check if the dimensions are correct
            if image_info['width'] != width or image_info['height'] != height:
                wrong_dimensions_count += 1
                image_info['width'] = width
                image_info['height'] = height
        else:
            print(f"Warning: Image file {image_filename} not found in {image_folder_path}")


    # Save the updated COCO JSON
    updated_json_path = os.path.splitext(coco_json_path)[0] + '_updated.json'
    with open(updated_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Number of images with wrong dimensions: {wrong_dimensions_count}")
    print(f"Updated JSON saved to: {updated_json_path}")

# Example usage
coco_json_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/annotations/instances_train2017.json"
image_folder_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/train2017"
update_image_dimensions(coco_json_path, image_folder_path)
