import cv2
import numpy as np
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes=cv2.CascadeClassifier('haarcascade_eye.xml')
smile=cv2.CascadeClassifier('haarcascade_smile.xml')
def detect(gray,frame):
    facevalues=face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in facevalues:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"Face_Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
        roi=gray[y:y+h,x:x+w]
        roicolor = frame[y:y + h , x:x + w]
        eyevalues=eyes.detectMultiScale(roi,1.1,22)
        for (ex,ey,ew,eh) in eyevalues:
            cv2.rectangle(roicolor,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            cv2.putText(roicolor , "Eye_Detected" , (ex , ey - 5) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255 , 0 , 0) , 1)
        smiles = smile.detectMultiScale(roi , 1.7 , 22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roicolor,(sx,sy),(sx+sw,sy+sh),(0,255,0),1)
            cv2.putText(roicolor , "Smile_Detected" , (sx , sy - 5) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255 , 255 , 0) , 1)
    return frame
cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Detection',canvas)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
