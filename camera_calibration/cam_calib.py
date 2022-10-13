import cv2
import numpy as np
import glob

## FIND CHESSBOARD CORNERS

chessboardSize = (9,6)
frameSize = (1280, 800)

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points. like (0,0,0), (1,0,0), (2,0,0), ...,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

#arrrays to store object points and image points from all the images.
objPoints = [] #3d points in the real world
imgPoints = [] #2d points in the image


images = glob.glob('camera_calibration/*.png')

for image in images:
    print(image)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #find chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    #if found, add object point, image points (after refining them)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        #draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)


cv2.destroyAllWindows()
    


################ CALIBRATION #################################################



ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

np.savez("CameraParams", cameraMatrix=cameraMatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Camera Calibrated: ", ret)
print("\nCamera Matriz:\n", cameraMatrix)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors\n", tvecs)



################ UNDISTORTION #####################################################


img = cv2.imread('camera_calibration/cali16.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


#undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#crop
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('Result1.png', dst)

#undistort with remapping
#mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
#dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#crop the image
#x, y, w, h, = roi
#dst = dst[y:y+h, x:x+h]
#cv2.imwrite('Result2.png', dst)


#repojection error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
    mean_error += error
    
print("\ntotal error: {}".format(mean_error/len(objPoints)))
print("\n\n\n")
