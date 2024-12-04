# import opencv
import cv2
import numpy as np
import mediapipe as mp
import math 
#Image Preprocessing

# Load the input image
image = cv2.imread('data\images\p2.jpeg')

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(image, (368, 716))

#Image Segmentation
masked_image = cv2.imread("data\\results\p2.png")
#Pixel Estimation

height_inp = int(input("Enter your height in cm : "))
# topPoint = 0
# bottomPoint = 0
# break_out_flag = False
# h, w, _ = masked_image.shape
# for row in range(len(masked_image)):
#     for col in range(len(masked_image[0])):
#         if(masked_image[row][col][0] > 0):
#             topPoint = row
#             break_out_flag = True
#             break
#     if break_out_flag:
#         break
# break_out_flag=False
# for row in range(len(masked_image)-1,-1,-1):
#     for col in range(len(masked_image[0])-1,-1,-1):
#         if(masked_image[row][col][0] > 0):
#             bottomPoint = row
#             break_out_flag = True
#             break
#     if break_out_flag:
#         break
# pixel = height_inp / (bottomPoint-topPoint)
# print("pixel_in_cm",pixel)

#Pose Estimation for linear measurements
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
img = masked_image
rgbIMG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img)
topPointx = 0
bottomPointx = 0
topPointY = 0
bottomPointY = 0
left_x = 0
right_x = 0
left_y = 0
right_y = 0
down_x = 0
down_y = 0
arm_x = 0
arm_y = 0
#print(results.pose_landmarks)
if results.pose_landmarks:
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        print(lm)
        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if(id==0):
            topPointx = cx
            topPointY = cy
        if(id==28):
            bottomPointx = cx
            bottomPointY = cy
        if(id==12):
            right_x= cx
            right_y= cy
        if(id==11):
            left_x= cx
            left_y= cy
        if(id==24):
            down_x= cx
            down_y= cy
        if(id==16):
            arm_x= cx
            arm_y= cy

        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
    pixel = height_inp / (math.sqrt((bottomPointx-topPointx)**2+(bottomPointY-topPointY)**2))
    print("pixel_in_cm",pixel)
    shoulderlen=math.sqrt((left_x-right_x)**2+(left_y-right_y)**2)*pixel
    print("sholderlen",shoulderlen)
    shirtlen=math.sqrt((down_y-right_y)**2+(down_x-right_x)**2)*pixel
    print("shirtlen",shirtlen)
    pantlen=math.sqrt((bottomPointY-down_y)**2+(bottomPointx-down_x)**2)*pixel
    print("pantlen",pantlen)
    armlen = math.sqrt((right_x-arm_x)**2+(right_y-arm_y)**2)*pixel
    print("armlen",armlen)

cv2.imshow("Image", img)
cv2.waitKey(0)


