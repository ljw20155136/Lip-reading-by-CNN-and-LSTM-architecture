import cv2
import numpy as np
import os

mouth_cascade = cv2.CascadeClassifier('./node-opencv-master/node-opencv-master/data/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')

for i in range(1, 101):
    cap = cv2.VideoCapture('./deepset/news1/%d.avi' % i)
    # recommend avi file, mp4 file often cause some error
    ret, frame = cap.read()
    j = 1
    ds_factor = 0.5
    height, width = frame.shape[:2]

    frame = cv2.resize(frame, (width, height), fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)

    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        break

    track_window = (x, y, w, h)
    roi = frame[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret2, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(roi, roi, mask=mask)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 255], [0, 180, 0, 255])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while (1):
        ret, frame = cap.read()
        if not os.path.exists('./dataset/News/S%d' % i):
            os.makedirs('./dataset/News/S%d' % i)
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 0, 1)
            lip = frame[y:y + h, x:x + w]
            video = cv2.resize(lip, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./dataset/News/S%d/%d.jpg" % (i, j), video)
            j = j + 1

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                continue

        else:
            break

cv2.waitKey(27)
cv2.destroyAllWindows()
cap.release()
