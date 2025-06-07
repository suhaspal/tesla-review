import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
with open('./model.p', 'rb') as file:
    loaded_model = pickle.load(file)
classifier = loaded_model['model']

# Initialize video capture on device 2
video_stream = cv2.VideoCapture(2)

# Set up MediaPipe Hands
mp_hands_module = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand_detector = mp_hands_module.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# Label mapping
gesture_labels = {0: 'A', 1: 'B', 2: 'L'}

while True:
    landmarks_features = []
    all_x = []
    all_y = []

    ret, img = video_stream.read()
    if not ret:
        break

    img_height, img_width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection = hand_detector.process(img_rgb)

    if detection.multi_hand_landmarks:
        for hand in detection.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand,
                mp_hands_module.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        for hand in detection.multi_hand_landmarks:
            for lm in hand.landmark:
                all_x.append(lm.x)
                all_y.append(lm.y)
            for lm in hand.landmark:
                landmarks_features.append(lm.x - min(all_x))
                landmarks_features.append(lm.y - min(all_y))

        x_min = int(min(all_x) * img_width) - 10
        y_min = int(min(all_y) * img_height) - 10
        x_max = int(max(all_x) * img_width) - 10
        y_max = int(max(all_y) * img_height) - 10

        pred = classifier.predict([np.asarray(landmarks_features)])
        gesture = gesture_labels[int(pred[0])]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
        cv2.putText(
            img, gesture, (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
        )

    cv2.imshow('Hand Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

video_stream.release()
cv2.destroyAllWindows()
