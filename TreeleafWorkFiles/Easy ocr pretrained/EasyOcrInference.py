import easyocr
import cv2
import matplotlib.pyplot as plt


reader = easyocr.Reader(['ne', 'en'])  


image_path = '3cb09b53-9a52-4037-9204-9ded694d52b0\plate263.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


result = reader.readtext(image)


for (bbox, text, prob) in result:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title('License Plate OCR Result')
plt.show()


print("Recognized Text:")
for (bbox, text, prob) in result:
    print(f"{text} (Confidence: {prob:.2f})")