import cv2
from my_hand_detection import HandDetection

hand_detection = HandDetection(min_detection_confidence=0.5, min_tracking_confidence=0.5)

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    status, frame = webcam.read()
    if not status:
        break

    frame = cv2.flip(frame, 1)
    hand_landmarks = hand_detection.findHandLandmarks(image=frame, draw=True)

    cv2.imshow("Hand Landmark", frame)
    if cv2.waitKey(1) == ord('h'):
        break

cv2.destroyAllWindows()
webcam.release()
