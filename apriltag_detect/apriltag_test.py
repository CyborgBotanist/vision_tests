from array import array
import cv2
import numpy as np
import apriltag as ap

#for networktables
from networktables import NetworkTables
#to see messages from networktables, you must setup logging
import logging


#Creating cam VideoCapture object to read USB camera
"""cam = cv2.VideoCapture(0)

#Loop while cam is opened.
while (cam.isOpened()):
    ret, frame = cam.read()

    #resize frame
    frame = cv2.resize(frame, (640, 400), fx = 0, fy = 0, 
                        interpolation = cv2.INTER_CUBIC)

    #display image
    cv2.imshow('Frame', frame)

    #necessary for thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #adaptive thresholding to use different threshold
    #values on different regions of the frame.
    Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    

    cv2.imshow('Thresh', Thresh)
    # define key "q" as exit key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

"""

#setting up networktables
logging.basicConfig(level=logging.DEBUG)


NetworkTables.initialize(server='127.0.0.2')

def valueChanged(table, key, value, isNew):
    print("valueChanged: key: '%s'; value: %s; isNew: %s" % (key, value, isNew))


def connectionListener(connected, info):
    print(info, "; Connected=%s" % connected)


NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

sd = NetworkTables.getTable("SmartDashboard")
sd.addEntryListener(valueChanged)
table = NetworkTables.getTable("ArrayVals")



distance = 0
rotation = 0
#translation in x and y are from center. upper left: (-,-), upper right: (+,-)
# lower right: (+,+), lower left: (-,+)
translation_x = (0,3)
translation_y = (1,3)
translation_z = (2,3)

tag_size = 0.146 #tag size in meters

frame_size = 1280, 800

options = ap.DetectorOptions(families="tag36h11", nthreads=1, quad_contours=False)
detector = ap.Detector(options)

image_path = 'photos/36h11tag.png'

#import camera params: 
#with np.load('camera_calibration/CameraParams.npz') as file:
#needed to run from terminal:
with np.load('~/py_ws/vision_tests/CameraParams.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]

# new camera matrix fx, fy, cx, cy
cam_params = mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]

def point_to_coordinate(frame=array, points=array):
    x = 1/frame[0] * points[0]
    y = 1/frame[1] * points[1]

    point_2_xy = x, y

    return point_2_xy


cam = cv2.VideoCapture(0)

while(cam.isOpened):
    #read image, convert to grayscale
    #image = cv2.imread(image_path)
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #show grayscale image
    cv2.imshow('gray', gray)

    #pass image to detector
    results = detector.detect(gray) 
    print("[INFO] {} total AprilTags detected".format(len(results)))
    

    #check for detections
    if len(results) >= 1:
        print("Apriltag detected")

        #detect pose of the tag
        detection_array = detector.detection_pose(results[0], cam_params, tag_size)
        pose = detection_array[0]
        
        #getting information from the detection. 
        #possible outputs: center, corner, tag_family, tag_id, 
        #DetectionBase, hamming, goodness, decision_margin, homography
        tf = results[0].tag_family
        tid = results[0].tag_id

        #send detection pose x, y, &z to networktables
        table.putNumber("TransX", pose[translation_x])
        table.putNumber("TransY", pose[translation_y])
        table.putNumber("TransZ", pose[translation_z])


        """print(tf)
        print(tid)
        print("decision margin: ")
        print(results[0].decision_margin)
        print("hamming: ")
        print(results[0].hamming)
        print("goodness")
        print(results[0].goodness)

        print("center of tag (x, y): ")
        print(point_to_coordinate(frame_size, results[0].center))

        print("camera matrix:")
        print(mtx[0,0])"""
        print(detection_array[0])
        print(pose[translation_x])




    k = cv2.waitKey(1500)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
