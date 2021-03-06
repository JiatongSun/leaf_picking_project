import time
import numpy as np
import cv2
import imageio

from drawBoundingRect import GetRect
from helpers import videorange

#See: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html#lucas-kanade-optical-flow-in-opencv

VIDEO = "test_video.mp4"
OUTPUT = "output.mp4"

# cap = cv2.VideoCapture(VIDEO)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

G = GetRect()

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
VR = videorange(VIDEO)
old_frame = next(VR)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
boxes = G.drawBoundingRect(old_frame) #shape: (#boxes, #points, 2)

# Use only one box
box = boxes[0]
#print(box)

# make mask for goodFeaturesToTrack
minx, miny = np.amin(box, axis=0)
maxx, maxy = np.amax(box, axis=0)
mask = np.zeros_like(old_gray)
mask[miny:maxy, minx:maxx] = 1

p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
p0 = p0[0]
print(p0.shape)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

modified_frames = []

for frame in VR:
    start_time = time.time()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                           p0, None, **lk_params)
    if p1 is None:
        break

    # Select good points
    good_new = p1[(st==1)[0]]
    good_old = p0[(st==1)[0]]

    if len(good_new) == 0:
        break

    elapsed_time = time.time() - start_time
    fps = str(int(1/elapsed_time)) if elapsed_time > 0 else "inf"

    x_new, y_new = good_new[0]
    x_old, y_old = good_old[0]
    
    minx, miny = minx+(x_new-x_old), miny+(y_new-y_old)
    maxx, maxy = maxx+(x_new-x_old), maxy+(y_new-y_old)

    # draw the tracks
    mask = cv2.line(mask, (x_new,y_new),(x_old,y_old), (255, 255, 0), 2)
    img = cv2.add(frame,mask)
    img = cv2.rectangle(img, (int(minx), int(miny)),
                (int(maxx), int(maxy)), color = (0, 255, 0), thickness = 2)
    img = cv2.putText(img, fps+" fps", (20,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
    modified_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.copy()

imageio.mimwrite(OUTPUT, modified_frames, fps=30)

cv2.destroyAllWindows()
