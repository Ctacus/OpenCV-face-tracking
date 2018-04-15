import numpy as np
import cv2
import random

from matplotlib import pyplot as plt

methods = [
    # {'color': (0, 0, 255), 'name': 'cv2.TM_CCOEFF'},
    {'color': (100, 150, 200), 'name': 'cv2.TM_CCOEFF_NORMED'},
    # {'color': (0, 255, 0), 'name': 'cv2.TM_CCORR'},
    # {'color': (0, 150, 0), 'name': 'cv2.TM_CCORR_NORMED'},
    # {'color': (255, 0, 0), 'name': 'cv2.TM_SQDIFF'},
    # {'color': (150, 0, 0), 'name': 'cv2.TM_SQDIFF_NORMED'}
]


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


def match(img, template, method, w, h, show_plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template[1], method)

    if show_plot:
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)

        plt.show()
    return get_boundaries(res, method, w, h)


def max_scaled_math(img, template, max_downscale, max_steps, method, show_plot):
    w, h = template[1].shape[::-1]
    step = (1 - max_downscale) / (max_steps - 1)
    scale = 1
    best = (0, 0, 0, 1)
    for n in range(1, max_steps):
        # TODO: сделать масштабирование по одному разу для каждого кадра
        copy = cv2.resize(img, None, fx=scale, fy=scale)
        t, b, m = match(copy, template, method, w, h, show_plot)
        if m > best[2]:
            best = (t, b, m, scale)
        scale = scale - step
    return best


def scale_coord(coord, scale):
    return tuple(round(c / scale) for c in coord)


def match_and_draw(img, templates, show_plot):
    scale = None
    for template in templates:
        w, h = template[1].shape[::-1]
        path = template[0]
        for meth in methods:
            # copy = img.copy()
            meth_name = meth['name']
            method = eval(meth_name)
            if scale is None:
                top_left, bottom_right, max_match, scale = max_scaled_math(img, template, 0.25, 8, method, show_plot)
            else:
                top_left, bottom_right, max_match = match(cv2.resize(img, None,  fx=scale, fy=scale), template, method, w, h, show_plot)
            if max_match > 0.53:
                rect(img, scale_coord(top_left, scale), scale_coord(bottom_right, scale), meth['color'],
                     ['{path} [{max:.3f}] x{scale:.2f}'.format(path=path, max=max_match, scale=scale), meth_name[7:]])

# первый шаблон - главный объект, все последующие - его части
template_paths = [r'me2.png', r'imgs/smile.png']
templates = [(path, cv2.imread(path, 0)) for path in template_paths]

cap = cv2.VideoCapture(0)

f = 0
rot = 0
cnd = True
plot = False
last_frame = None
step = 1
max_buff = 30
buff = []
shadow = False
while cnd:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here


    # Display the resulting frame
    # b, g, r = cv2.split(frame)
    # c = random.choice(range(5))
    # frame[:, :, 2] = 0

    if shadow:
        frame_copy = frame.copy()
        if len(buff) >= step and f >= step:
            diff = cv2.subtract(frame_copy, buff[(f - step) % max_buff])
            cv2.putText(diff, '{}'.format(step), (10, 25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (50, 50, 200), 2)
            cv2.imshow('diff', diff)
        if f < max_buff:
            buff.append(frame_copy)
        else:
            buff[f % max_buff] = frame_copy
    f = f + 1
    match_and_draw(frame, templates, plot)
    cv2.imshow('frame', frame)

    # cnd = False
    key = cv2.waitKey(3) & 0xFF

    if key == ord('q'):
        break
    # plot = cv2.waitKey(1) & 0xFF == ord('p') and not plot

    elif key == ord('+'):
        step = min(max_buff, step + 1)
    elif key == ord('-'):
        step = max(1, step - 1)
    elif key == ord('s'):
        f = 0
        shadow = not shadow

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#TODO: 1. выделение границ движущихся объектов
#TODO: 2. улучшить производительности при работе с большим количеством шаблонов
