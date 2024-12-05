import pytesseract
from PIL import Image

# Specify the path to the Tesseract executable (if not added to PATH)
# For example, on Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)

    # Clean up the text (remove extra spaces, newlines, etc.)
    cleaned_text = text.strip()

    return cleaned_text

# Specify the image path
image_path = 'Capture.jpeg'

# Perform OCR on the image
extracted_text = perform_ocr(image_path)

# Print the final extracted text
print("Final Extracted Text:", extracted_text)
