import cv2
import numpy as np

video = cv2.VideoCapture('JurassicPark.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if (video.isOpened() == False):
    print('Erro ao abrir o arquivo de vídeo!')

while(video.isOpened()):
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

    for x, y, w, h in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 4)

    cv2.imshow('Frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
