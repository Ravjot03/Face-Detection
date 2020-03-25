# OpenCV program to detect face in real time 
# import libraries of python OpenCV  
# where its functionality resides

import cv2

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images.

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture frames from a camera 

cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.

while True:
    # reads frames from a camera 
    ret, frame = cap.read()
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
 
    # Detects faces of different sizes in the input image 
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30,30))
 
    for (x,y,w,h) in faces:
            cv2.circle(frame,(x+int(w/2),y+int(h/2)), 100, (0,0,255), 3)
 
     # Display an image in a window 
    cv2.imshow('camera', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Close the window 
cap.release()

# De-allocate any associated memory usage 
cv2.destroyAllWindows()
