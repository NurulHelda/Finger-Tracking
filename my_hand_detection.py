import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandDetection:
    def __init__(self, max_num_hand=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(max_num_hands=max_num_hand, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandmarks(self, image, handNumber=0, draw=False):
        original_image = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        landmark_list = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(hand.landmark):
                img_h, img_w, img_c = original_image.shape
                x_pos, y_pos = int(landmark.x * img_w), int(landmark.y * img_h)
                landmark_list.append([id, x_pos, y_pos])
            if draw:
                mp_draw.draw_landmarks(original_image, hand, mp_hands.HAND_CONNECTIONS)

        return landmark_list
    