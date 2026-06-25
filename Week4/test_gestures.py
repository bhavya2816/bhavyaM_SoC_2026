import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

import gestures


def draw_landmarks(img, data):

    h, w, _ = img.shape

    points = []

    for lm in data:

        x = int(lm.x * w)
        y = int(lm.y * h)

        points.append((x, y))

        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for s, e in connections:
        cv2.line(img, points[s], points[e], (255,255,255), 2)


def get_landmark_list(img, hand):

    h, w, _ = img.shape

    lm_list = []

    for idx, lm in enumerate(hand):

        x = int(lm.x * w)
        y = int(lm.y * h)

        lm_list.append([idx, x, y])

    return lm_list


model_path = "models/hand_landmarker.task"

base = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)

prev_time = 0

while True:

    ok, img = cam.read()

    if not ok:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    gesture = "NONE"

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        hand_label = "Right"

        if result.handedness:

            hand_label = result.handedness[0][0].display_name

            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"

        draw_landmarks(img, hand)

        lm_list = get_landmark_list(img, hand)

        gesture = gestures.classify(
            lm_list,
            hand_label
        )

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time) if prev_time else 0

    prev_time = curr_time

    cv2.putText(
        img,
        f"Gesture : {gesture}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.putText(
        img,
        f"FPS : {int(fps)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.imshow("Gesture Test", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()