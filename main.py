import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret = True

paintWindow = np.zeros((1000, 1500, 3), dtype=np.uint8) + 255

drawing_color = (0, 0, 0)  # Black by default
drawing_thickness = 5

prev_index_tip = (0, 0)
prev_thumb_tip = (0, 0)

drawing = False

while ret:

    ret, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        index_tip = (int(landmarks[8].x * 1500), int(landmarks[8].y * 1000))
        thumb_tip = (int(landmarks[4].x * 1500), int(landmarks[4].y * 1000))

        distance = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))

        if distance < 50:
            paintWindow = np.zeros((1000, 1500, 3), dtype=np.uint8) + 255
        else:
            drawing = True
            mpDraw.draw_landmarks(frame, result.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS,
                                  drawing_color, drawing_thickness)

            if prev_index_tip != (0, 0) and drawing:
                cv2.line(paintWindow, prev_index_tip, index_tip, drawing_color, drawing_thickness)

            prev_index_tip = index_tip
            prev_thumb_tip = thumb_tip

        drawing = False
        prev_index_tip = (0, 0)
        prev_thumb_tip = (0, 0)

    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()