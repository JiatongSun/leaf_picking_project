import cv2
import numpy as np
from helpers import videorange

class GetRect():
    def __init__(self):
        self.initCoords = []
        self.mInitCoords = None
        self.mFinalCoords = None
        self.finalCoords = []
        self.isDrawing = False

    def drawBoundingRect(self, img):
        self.initCoords = []
        self.finalCoords = []
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self._getCoords, param=img)

        while True:
            if not self.isDrawing or self.mInitCoords:
                clone = img.copy()
                for i,f in zip(self.initCoords, self.finalCoords):
                    cv2.rectangle(clone,(i[0],i[1]),(f[0],f[1]),(0,255,0),1)
                cv2.rectangle(clone,self.mInitCoords,self.mFinalCoords,(0,255,0),1)
                cv2.imshow("image", clone)
            else:
                cv2.imshow("image", img)
            k = cv2.waitKey(10)
            #q to quit execution
            if k == ord('q'):
                die("quitted")
            #c to quit from current frame
            if k == ord('c'):
                cv2.destroyAllWindows()
                break

        #convert {topleft + bottomright repr} to {4 point repr}
        return np.array([(i,(f[0],i[1]),f,(i[0],f[1])) for i,f in zip(self.initCoords, self.finalCoords)])

    def _getCoords(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.isDrawing = True
            self.mInitCoords = (x,y)
            self.mFinalCoords = (x,y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.isDrawing:
                self.mFinalCoords = (x,y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.isDrawing = False
            self.initCoords.append(self.mInitCoords)
            self.finalCoords.append(self.mFinalCoords)

if __name__ == "__main__":
    g = GetRect()
    for frame in videorange('video1.avi'):
        points = g.drawBoundingRect(frame)
        print(points)
        break
