# List of characters corresponding to class indices 0-93
characters = [
    "अ", "२", "आ", "ं", "इ", "ई", "३", "ि", "उ", "ऊ", "४", "ऋ", "ऌ", "ए", "५", 
    "ा", "ऐ", "ओ", "औ", "अं", "६", "अः", "बा", "लु", "७", "प्र", "ती", "दे", "ना", 
    "ृ", "भे", "८", "क", "ख", "ै", "ग", "घ", "ङ", "९", "च", "छ", "ज", "ी", "झ", 
    "ञ", "ः", "ट", "ठ", "े", "ड", "ु", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", 
    "फ", "ब", "ू", "भ", "म", "य", "र", "ल", "व", "श", "ष", "ौ", "स", "ह", "क्ष", 
    "त्र", "ज्ञ", "बा", "लु", "ृ", "प्र", "ती", "दे", "ना", "भे", "०", "१", "बा", 
    "लु", "प्र", "ती", "दे", "ना", "भे", "ो"
]

# Function to convert labels
def convert_labels(input_file, output_file, page_number):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines
            
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)
            
            if class_id < 0 or class_id >= len(characters):
                continue  # Skip invalid class IDs
            
            # Convert YOLO format to the specified format
            char = characters[class_id]
            left = x_center - width / 2
            right = x_center + width / 2
            bottom = y_center - height / 2
            top = y_center + height / 2
            
            # Write the converted label
            outfile.write(f"{char} {left:.6f} {bottom:.6f} {right:.6f} {top:.6f} {page_number}\n")

# Input and output file paths
input_file = "character.v1i.yolov8/train/labels/train_text_png.txt"  # Replace with your input file
output_file = "character.v1i.yolov8/train/labels/output.txt"  # Replace with your output file
page_number = 1  # Specify the page number

# Convert labels
convert_labels(input_file, output_file, page_number)

print("Labels converted and saved to", output_file)
