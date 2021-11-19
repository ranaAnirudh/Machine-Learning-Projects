import cv2
from random import randrange
# print(cv2.__version__)
trained_face_data=cv2.CascadeClassifier('haarcascades_frontalface_default.xml')
#choose an image
# img=cv2.imread('group.jpg')

webcam=cv2.VideoCapture(0)
#run all over the all frames
while True:
    # reads the current frames
    successful_frame_read,frame=webcam.read()

    # must convert to grayscale
    gray_scale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect image of any size == MULTiSCale
    face_coordinates=trained_face_data.detectMultiScale(gray_scale_img)
    # print(face_coordinates)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y), (x+w, y+h ), (randrange(128,255),randrange(255),randrange(255)), 4)


    cv2.imshow('Face detector',frame)
    #to hold the image until a key pressed
    key=cv2.waitKey(1) 
    # dtop if Q is pressed
    if key==81 or key==113:
        break
webcam.release()
print('Code Completed!')