import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui  # 👈 Bas is line ka import extra

# Load transparent hand template
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]

# Open webcam
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8)

# Function to overlay PNG with alpha
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c] = (
            alpha_overlay * img_overlay[:, :, c] +
            alpha_background * img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c]
        )

# Variables to control triggering
hand_inside_start = None
video_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    center_x = w // 2 - template_w // 2
    center_y = h // 2 - template_h // 2

    # Overlay template
    overlay_image_alpha(frame, template, (center_x, center_y))

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_inside = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Check if hand bounding box inside template
            if (x_min > center_x and x_max < center_x + template_w and
                y_min > center_y and y_max < center_y + template_h):
                hand_inside = True
                if hand_inside_start is None:
                    hand_inside_start = time.time()

                elif (time.time() - hand_inside_start > 1.5) and not video_playing:
                    video_playing = True
                    video = cv2.VideoCapture('vid.mp4')

                    # Get screen resolution 👇
                    screen_width, screen_height = pyautogui.size()

                    while video.isOpened():
                        ret_vid, frame_vid = video.read()
                        if not ret_vid:
                            break

                        # Auto-resize video to screen size
                        frame_vid = cv2.resize(frame_vid, (screen_width, screen_height))
                        cv2.imshow('Hand Scanner', frame_vid)
                        if cv2.waitKey(25) & 0xFF == 27:
                            break
                    video.release()
                    video_playing = False
                    hand_inside_start = None
            else:
                hand_inside = False

    if not hand_inside:
        hand_inside_start = None

    # Show webcam frame with overlay
    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
