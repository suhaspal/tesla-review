import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_PATH = './data'

features = []
targets = []

# Loop through each label directory
for label in os.listdir(DATA_PATH):
    label_folder = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_folder):
        continue
    # Loop through each image in the label directory
    for file_name in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file_name)
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        detection = hand_detector.process(img_rgb)
        if detection.multi_hand_landmarks:
            for hand in detection.multi_hand_landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                min_x, min_y = min(xs), min(ys)
                normalized = []
                for lm in hand.landmark:
                    normalized.extend([lm.x - min_x, lm.y - min_y])
                features.append(normalized)
                targets.append(label)

# Save the dataset
with open('data.pickle', 'wb') as out_file:
    pickle.dump({'data': features, 'labels': targets}, out_file)
