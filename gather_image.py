import os
import cv2

BASE_PATH = './data'
CLASS_COUNT = 3
IMAGES_PER_CLASS = 100

# Ensure the base directory exists
os.makedirs(BASE_PATH, exist_ok=True)

# Initialize webcam (change index as needed)
camera = cv2.VideoCapture(2)

for class_idx in range(CLASS_COUNT):
    class_dir = os.path.join(BASE_PATH, str(class_idx))
    os.makedirs(class_dir, exist_ok=True)
    print(f"Starting collection for category {class_idx}")

    # Wait for user to get ready
    while True:
        grabbed, img = camera.read()
        if not grabbed:
            print("Camera frame not received. Exiting.")
            break
        cv2.putText(img, 'Ready? Press Q!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow('Capture Window', img)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

    # Capture images for current class
    saved = 0
    while saved < IMAGES_PER_CLASS:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame.")
            continue
        cv2.imshow('Capture Window', frame)
        cv2.waitKey(25)
        filename = os.path.join(class_dir, f'{saved}.jpg')
        cv2.imwrite(filename, frame)
        saved += 1

camera.release()
cv2.destroyAllWindows()
