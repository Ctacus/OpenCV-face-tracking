import numpy as np
import cv2
import random

from matplotlib import pyplot as plt


def rect(img, top_left, bottom_right, color, texts):
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    bottom_margin = 15
    spacing = 15
    i = 0
    for text in texts:
        cv2.putText(img, text, (top_left[0], bottom_right[1] + bottom_margin + spacing * i), cv2.FONT_HERSHEY_PLAIN, 1,
                    color, 2)
        i = i + 1


def get_boundaries(match_res, method, w, h):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, max_val


def match(img, templates):
    img2 = img.copy()

    # All the 6 methods for comparison in a list
    methods = [
        # {'color': (0, 0, 255), 'name': 'cv2.TM_CCOEFF'},
        {'color': (100, 150, 200), 'name': 'cv2.TM_CCOEFF_NORMED'},
        # {'color': (0, 255, 0), 'name': 'cv2.TM_CCORR'},
        # {'color': (0, 150, 0), 'name': 'cv2.TM_CCORR_NORMED'},
        # {'color': (255, 0, 0), 'name': 'cv2.TM_SQDIFF'},
        # {'color': (150, 0, 0), 'name': 'cv2.TM_SQDIFF_NORMED'}
    ]
    for template in templates:
        w, h = template[1].shape[::-1]
        path = template[0]
        for meth in methods:
            # copy = img.copy()
            meth_name = meth['name']
            method = eval(meth_name)
            color = meth['color']
            # Apply template Matching
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(gray, template[1], method)
            top_left, bottom_right, max = get_boundaries(res, method, w, h)

            if max > 0.5:
                rect(img, top_left, bottom_right, color,
                     ['{path} [{max:.3f}]'.format(path=path, max=max), meth_name[7:]])


template_paths = [r'me2.png']
templates = [(path, cv2.imread(path, 0)) for path in template_paths]

cap = cv2.VideoCapture(0)

f = 0
rot = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here


    # Display the resulting frame
    # b, g, r = cv2.split(frame)
    # c = random.choice(range(5))
    # frame[:, :, 2] = 0
    f = f + 1
    if f % 40 == 0:
        rot = (rot + 1) % 4

    # frame = np.rot90(frame, -rot)
    match(frame, templates)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()