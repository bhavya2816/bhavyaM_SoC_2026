import math

def normalized_distance(arr, a, b):

    return distance(arr, a, b) / hand_size(arr)
def distance(arr, a, b):

    x1 = arr[a][1]
    y1 = arr[a][2]

    x2 = arr[b][1]
    y2 = arr[b][2]

    return math.hypot(x2 - x1, y2 - y1)


def hand_size(arr):

    d = distance(arr, 0, 9)

    if d == 0:
        return 1

    return d


def fingers_up(arr, side="Right"):

    finger = []

    if side == "Right":

        if arr[4][1] > arr[3][1]:
            finger.append(1)
        else:
            finger.append(0)

    else:

        if arr[4][1] < arr[3][1]:
            finger.append(1)
        else:
            finger.append(0)

    for i in [8, 12, 16, 20]:

        if arr[i][2] < arr[i - 2][2]:
            finger.append(1)
        else:
            finger.append(0)

    return finger

def classify(lm_list, hand_label="Right"):
    """Returns the gesture name."""

    if len(lm_list) < 21:
        return "NONE"

    fingers = fingers_up(lm_list, hand_label)

    pinch = normalized_distance(lm_list, 4, 8)

    # Distance-based gesture first
    if pinch < 0.30 and fingers[2] and fingers[3] and fingers[4]:
        return "OK"

    # Finger-count gestures
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    if fingers == [0, 1, 0, 0, 0]:
        return "POINT"

    if fingers == [0, 1, 1, 0, 0]:
        return "CHEERS"

    if fingers == [1, 0, 0, 0, 0]:
        return "THUMB"
    if fingers == [1,1, 0, 0, 1]:
        return "SPIDERMAN"

    return "UNKNOWN"

