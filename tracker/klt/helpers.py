import numpy as np
import cv2
import sys

def die(msg=""):
  print(msg)
  sys.exit()

#https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  if image.ndim == 3:
    (h, w, _) = image.shape
  else:
    (h, w) = image.shape
  dim = None
  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))

  return cv2.resize(image, dim, interpolation=inter)

#s = scale factor
def disp(img, s=1, wait=True):
  resized = ResizeWithAspectRatio(img, width=int(img.shape[1]*s))
  normalized = cv2.normalize(resized, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
  cv2.imshow('img', normalized)
  if wait:
    cv2.waitKey(0)

"""
Iterator for reading videos (single frame)
Usage: for frame in videorange("videoname.mp4"):
"""
class videorange:
  #constructor
  def __init__(self, name):
    self.vid = cv2.VideoCapture(name)
    if not self.vid.isOpened():
      die("Error opening video")

  #initialize iter
  def __iter__(self):
    return self

  #get next value
  def __next__(self):
    if not self.vid.isOpened():
      self.vid.release()
      raise StopIteration
    ret, frame = self.vid.read()
    if not ret:
      self.vid.release()
      raise StopIteration
    return frame