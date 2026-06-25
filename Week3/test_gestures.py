import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import time


def draw_landmarks(img, data):

    h, w, c = img.shape

    points = []

    for a in data:

        x = int(a.x * w)
        y = int(a.y * h)

        points.append((x, y))

        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    lines = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for s, e in lines:
        cv2.line(img, points[s], points[e], (255, 0, 0), 2)


def get_landmark_list(img, data):

    h, w, x = img.shape

    array = []

    n = 0

    for a in data:

        xx = int(a.x * w)
        yy = int(a.y * h)

        array.append([n, xx, yy])

        n = n + 1

    return array

# check which fingers are up
def fingers_up(lm_list):

    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (works for mirrored right hand)
    if lm_list[4][1] > lm_list[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:

        if lm_list[tip][2] < lm_list[tip - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# Recognize gesture

def classify_gesture(fingers):

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST BUMP"

    elif fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    elif fingers == [0, 1, 0, 0, 0]:
        return "YOUU!!"

    elif fingers == [0, 1, 1, 0, 0]:
        return "CHEERS"

    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"
    elif fingers == [1,1, 0, 0, 1]:
        return "SPIDERMAN"

    else:
        return "UNKNOWN"

model = "models/hand_landmarker.task"

base = python.BaseOptions(model_asset_path=model)

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

if cam.isOpened() == False:
    print("Camera not opened")
    exit()

while True:

    ok, img = cam.read()

    if ok == False:
        print("No frame")
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    t = int(time.time() * 1000)

    ans = detector.detect_for_video(image, t)

    if ans.hand_landmarks:

        for hand_landmarks in ans.hand_landmarks:

            # Draw skeleton
            draw_landmarks(img, hand_landmarks)

            # Get landmark coordinates
            lm_list = get_landmark_list(img, hand_landmarks)

            if lm_list:

              # Highlight index fingertip
                cv2.circle(
                    img,
                    (lm_list[8][1], lm_list[8][2]),
                    8,
                    (0, 0, 0),
                    -1
                )
                fingers = fingers_up(lm_list)
                gesture = classify_gesture(fingers)

                # Print gesture
                print(fingers)

                # Show gesture on screen
                cv2.putText(
                    img,
                    gesture,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    3
                )
    cv2.imshow("Hand Tracking", img)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
