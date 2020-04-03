import cv2
import numpy as np
import copy
import os
import keyboard
    
def crop_contour(points):
    print("In crop contour")
    new_list = []
    for elem in points:
        new_list.append(np.array(elem))
    new_list = np.array(new_list)
    mask = np.zeros_like(image)
    cv2.drawContours(mask,[new_list],-1,(255,255,255),-1)
    final = cv2.add(image,cv2.bitwise_not(mask))
    mask = mask[:,:,0]
    cv2.imwrite(os.path.join(mask_folder,'mask'+filename),mask)

    y,x,c = np.where(final != 255)
    TL_y,TL_x = np.min(y),np.min(x)
    BR_y,BR_x = np.max(y),np.max(x)
    
    cropped = final[max(TL_y-20,0):min(BR_y+20,image.shape[0]),
                    max(TL_x-20,0):min(BR_x+20,image.shape[1])]
    cv2.imwrite(os.path.join(crop_folder,'crop'+filename),cropped)
    cv2.imshow("cropped",cropped)
    
def click_and_crop(event, x, y, flag, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        
        if len(points) >=2 :
            cv2.line(image,points[-1],points[-2],(0,0,0),1)
            cv2.imshow(filename,image)
        if len(points) == 20:
            crop_contour(points)
            cv2.imshow(filename,image)

if __name__ == '__main__':
    mask_folder = 'train/mask'
    crop_folder = 'train/crop'
    if os.path.isdir(mask_folder) is False: os.mkdir(mask_folder)
    if os.path.isdir(crop_folder) is False: os.mkdir(crop_folder)
    for filename in os.listdir("train/origin"):
        filename = '53.jpg'
        points = []
        cut_flag = False
        image = cv2.imread(os.path.join("train/origin",filename))
        cv2.imshow(filename,image)
        cv2.setMouseCallback(filename,click_and_crop)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()