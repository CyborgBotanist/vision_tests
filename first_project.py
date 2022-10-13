import cv2
import sys
import apriltag
import networktables as NetworkTables
#import argparse
import numpy as np

image_w = 3024
image_h = 4032

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
imagepath = 'photos/april_tag.jpg'
imgpath ='photos/bengal_image.jpg'
img = cv2.imread('photos/bengal_image.jpg')


#image = cv2.imread(imagepath)
image_gray = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

image_gray_scaled = cv2.resize(image_gray, (756, 1008))


if image_gray is None:
    sys.exit("Could not read the image")

if img is None:
    sys.exit("Could not read the img")


#cv2.imshow('cat', img)
cv2.imshow('tag', image_gray_scaled)

detected = detector.detect(image_gray)


print("networktables version: " + NetworkTables.__version__)

print("opencv version: " + cv2.__version__)

cv2.waitKey(0)