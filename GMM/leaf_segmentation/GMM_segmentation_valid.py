import numpy as np
import cv2
import os

def gaussian(data,mean,cov):
        cov_inv = np.linalg.inv(cov)
        diff = np.matrix(data-mean)
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        return N

if __name__ == '__main__':
    result_folder = 'valid/result'
    contour_folder = 'valid/contour'
    if os.path.isdir(result_folder) is False: os.mkdir(result_folder)
    if os.path.isdir(contour_folder) is False: os.mkdir(contour_folder)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('valid/video.mp4',fourcc, 1, (1280,960),True)
    for filename in os.listdir('valid/origin'):
        test_image = cv2.imread(os.path.join('valid/origin',filename))
        K = 4
        nx, ny, ch = test_image.shape
        img = np.reshape(test_image, (nx*ny,ch))
        weights = np.load('weights.npy')
        parameters = np.load('parameters.npy',allow_pickle=True)
        prob = np.zeros((nx*ny,K))
        likelihood = np.zeros((nx*ny,K))
        for cluster in range(K):
                prob[:,cluster:cluster+1] = weights[cluster]*gaussian(img,parameters[cluster]['mean'], parameters[cluster]['cov'])
                likelihood = prob.sum(1)
        probabilities = np.reshape(likelihood,(nx,ny))
    
        valid_index = probabilities>np.max(probabilities)/12
        probabilities[valid_index == True] = 255
        probabilities[valid_index == False] = 0
        probabilities = probabilities.astype(np.uint8)
        probabilities3D = np.zeros_like(test_image)
        for depth in range(3):
            probabilities3D[:,:,depth] = probabilities
        gray = probabilities.copy()
        
        ys, xs = np.where(valid_index>0)
        sigma = np.sqrt(np.sum((ys-ys.mean())**2+(xs-xs.mean())**2)/len(xs))
        bright_ratio = probabilities.sum()/(255*nx*ny)
        print(f'{filename}: sigma: {sigma}, bright ratio: {bright_ratio}')
        if bright_ratio < 0.05 and sigma < 100:
            kernel = np.ones((8,8), np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=4)
            
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_contours = np.zeros_like(test_image)
        result = np.zeros_like(test_image)
        sort = sorted(contours, key=cv2.contourArea, reverse=True)
        cmax = sort[0]
        
        cv2.drawContours(img_contours, [cmax], -1, (0,255,0), 3)
        cv2.drawContours(result, [cmax], -1, (255,255,255), cv2.FILLED)
        
        # cv2.imshow('origin',cv2.resize(test_image,(320,240)))
        # cv2.imshow('likelihood',cv2.resize(probabilities3D,(320,240)))
        # cv2.imshow('contour', cv2.resize(img_contours,(320,240)))
        # cv2.imshow('result',cv2.resize(result,(320,240)))
        cv2.imwrite(os.path.join(contour_folder,'contour'+filename),img_contours)
        cv2.imwrite(os.path.join(result_folder,'result'+filename),result)
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (350, 450)
        fontScale = 2
        color = (0, 0, 255) 
        thickness = 3
        image1 = cv2.putText(test_image, 'origin', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        image2 = cv2.putText(probabilities3D, 'likelihood', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        image3 = cv2.putText(img_contours, 'contour', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        image4 = cv2.putText(result, 'result', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        vis_t = np.concatenate((image1, image2), axis=1)
        vis_b = np.concatenate((image3, image4), axis=1)
        vis = np.concatenate((vis_t, vis_b), axis=0)
        cv2.imshow('vis',cv2.resize(vis,(640,480)))
        video.write(vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    video.release()
    cv2.destroyAllWindows()