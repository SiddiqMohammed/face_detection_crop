import cv2
import os
import numpy as np 
from PIL import Image, ImageDraw 
  
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(1)

counter = 0
current_path = os.getcwd()


while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cX = x + w // 2
        cY =  y + h // 2
        center_coordinates = cX, cY
        radius = w // 2 
        cv2.circle(img, center_coordinates, radius, (0, 0, 100), 3)

        if counter == 20:

            rectX = (cX - radius) 
            rectY = (cY - radius)
            
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite(current_path + "\\facedetection" +
                        str(counter)+".png", crop_img)
            print("Crop Saved")


            n_img=Image.open("facedetection20.png") 

            height,width = n_img.size 
            lum_img = Image.new('L', [height,width] , 0) 

            draw = ImageDraw.Draw(lum_img) 
            draw.pieslice([(0,0), (height,width)], 0, 360, 
                        fill = 255, outline = "white") 
            img_arr =np.array(n_img) 
            lum_img_arr =np.array(lum_img) 
            final_img_arr = np.dstack((img_arr,lum_img_arr)) 
            Image.fromarray(final_img_arr).save("c.png")
            

        counter = counter + 1

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
