import os
import sys
from PIL import Image

def rename_images(directory, fontname):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Get all image files in the directory
    image_extensions = ('.png', '.jpg', '.jpeg', 'tif')
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

    # Sort the files to ensure consistent numbering
    image_files.sort()

    for i, filename in enumerate(image_files):
        # Construct the new filename
        new_filename = f"nep.{fontname}.exp{i}.tif"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Open the image and convert to TIFF if necessary
        with Image.open(old_path) as img:
            # If the image is not already TIFF, convert it
            if img.format != 'tif':
                img.save(new_path, 'TIFF', compression='tiff_deflate')
                os.remove(old_path)  # Remove the original file
                print(f"Converted and renamed: {filename} -> {new_filename}")
            else:
                # If it's already TIFF, just rename
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_images.py <directory> <fontname>")
        sys.exit(1)

    directory = sys.argv[1]
    fontname = sys.argv[2]
    rename_images(directory, fontname)
