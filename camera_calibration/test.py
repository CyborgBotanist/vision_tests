import numpy as np
import cv2
import glob
import apriltag

chessboardSize = (9,6) # chessboardSize[0] chessboardSize[1]
frameSize = (1280, 800)

image_path = ('photos/cali0.png')


######### UNDISTORT ################################################################

with np.load('camera_calibration/CameraParams.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]



def undistort(img):
    #undistort
    dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix)
    #crop
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('Result1.png', dst)
    return(dst)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


while True:
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    undistIMG = undistort(img)
    cv2.imshow('window     image path: ' + image_path, undistIMG)
    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break  



cv2.destroyAllWindows()