import cv2
#import opencv-python as cv2
import numpy as np 
from keras.models import load_model
import argparse
from PIL import Image
import imutils


def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples

    return mean_distance

# model=load_model('saved_model.h5')
#model=load_model('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\saved_model.h5')
model=load_model('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\saved_model.h5py')
# cap = cv2.VideoCapture("__path_to_custom_test_video")
cap = cv2.VideoCapture("C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\Avenue_Dataset\\Avenue Dataset\\testing_videos\\09.avi")
print(cap.isOpened())

while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()


    for i in range(10):
        ret,frame=cap.read()
        
        if frame is None:
            print('Wrong path:', cap)
        else:
            image = imutils.resize(frame,width=1000,height=1200)
            frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
            gray=(gray-gray.mean())/gray.std()
            gray=np.clip(gray,0,1)
            imagedump.append(gray)

    imagedump=np.array(imagedump)

    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)

    output=model.predict(imagedump)

    loss=mean_squared_loss(imagedump,output)
    print(loss)
    if frame.any()==None:
        print("none")
	

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    #if loss>0.00068:
    if loss>0.00041:
        print('Abnormal Event Detected')
        cv2.putText(image,"Abnormal Event",(220,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

    cv2.imshow("video",image)

cap.release()
cv2.destroyAllWindows()

