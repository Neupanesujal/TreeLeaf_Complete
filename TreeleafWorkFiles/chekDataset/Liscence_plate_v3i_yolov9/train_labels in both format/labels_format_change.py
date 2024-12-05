import os

# Path for the YOLO annotation files
annotation_folder = '/home/sujal-neupane/drives/HDD/part1/TreeleafWorkFiles/chekDataset/Liscence_plate_v3i_yolov9/train/labels'  # Folder where your 7 .txt files are located
image_size = 244  # since your images are 244x244

# Mapping for class tag to Devanagari digits and specific characters
class_mapping = {
    0: '०', 1: '१', 2: '२', 3: '३', 4: '४', 5: '५', 
    6: '६', 7: '७', 8: '८', 9: '९', 
    10: 'बा', 11: 'प'
}

# Function to convert YOLOv9 format to required format
def convert_yolo_to_bbox(yolo_data, img_size):
    class_tag, x_center, y_center, width, height = map(float, yolo_data.split())
    
    # Calculate left, right, top, bottom
    left = (x_center - width / 2) * img_size
    right = (x_center + width / 2) * img_size
    top = (y_center - height / 2) * img_size
    bottom = (y_center + height / 2) * img_size
    
    return int(class_tag), left, bottom, right, top

# Iterate through all txt files
for img_index in range(0, 7):  # Assuming 7 images and 7 txt files
    txt_file = os.path.join(annotation_folder, f'{img_index}.txt')
    
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        converted_annotations = []
        
        # Convert each annotation in the txt file
        for line in lines:
            class_tag, left, bottom, right, top = convert_yolo_to_bbox(line, image_size)
            
            # Map class_tag to Devanagari or specific characters
            class_character = class_mapping.get(class_tag, 'Unknown')

            # Append converted annotation with <page> set to 0
            converted_annotations.append(f"{class_character} {left:.6f} {bottom:.6f} {right:.6f} {top:.6f} 0")
        
        # Save the converted format into a new txt file
        output_file = os.path.join(annotation_folder, f'converted_image_{img_index}.txt')
        with open(output_file, 'w') as file:
            file.write("\n".join(converted_annotations))
