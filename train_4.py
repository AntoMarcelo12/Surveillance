
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
import cv2
from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils


def store_inarray(image_path):
    image=load_img(image_path)
    image=img_to_array(image)
    image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
    gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
    store_image.append(gray)





videos=os.listdir('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\Avenue_Dataset\\Avenue Dataset\\training_videos\\')
contatore=0
count = 0
store_image=[]
for video in videos:
  #video
  vidcap = cv2.VideoCapture('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\Avenue_Dataset\\Avenue Dataset\\training_videos\\'+video)
  # contatore=contatore + 1
  # contatore
  #vidcap = cv2.VideoCapture('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\Avenue_Dataset\\Avenue Dataset\\training_videos\\01.avi')
  success,image = vidcap.read()
  
  
  
  while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file   
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file 
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
    # store_inarray(vidcap)
    if image is None:
      print('Wrong path:', vidcap)
    else:
      image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
      gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
      store_image.append(gray)
      
  # vidcap = cv2.VideoCapture('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\Avenue_Dataset\\Avenue Dataset\\training_videos\\02.avi')
  # success,image = vidcap.read()
#count = 0
#store_image=[]

# while success:
#   #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file   
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file 
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1
#   # store_inarray(vidcap)
#   if image is None:
#     print('Wrong path:', vidcap)
#   else:
#     image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
#     gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
#     store_image.append(gray)
#   
count


import gc
gc.collect()
gc.enable()


store_image=np.array(store_image)
a,b,c=store_image.shape
store_image.resize(b,c,a)
store_image=(store_image-store_image.mean())/(store_image.std())
store_image=np.clip(store_image,0,1)
np.save('training_new.npy',store_image)


stae_model=Sequential()
stae_model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
stae_model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
stae_model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))
stae_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


training_data=np.load('training_new.npy')
frames=training_data.shape[2]
frames=frames-frames%10
training_data=training_data[:,:,:frames]
training_data=training_data.reshape(-1,227,227,10)
training_data=np.expand_dims(training_data,axis=4)
target_data=training_data.copy()
epochs=5
batch_size=1
# callback_save = ModelCheckpoint("saved_model.h5", monitor="mean_squared_error", save_best_only=True)
# callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#callback_save = ModelCheckpoint('saved_model.h5', monitor='loss', save_best_only=True)
callback_save = ModelCheckpoint('saved_model.h5py', monitor='loss', save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='loss', patience=3)
hist=stae_model.fit(training_data,target_data, batch_size=batch_size, epochs=epochs, callbacks = [callback_save,callback_early_stopping])
# stae_model.fit(training_data,target_data, batch_size=batch_size, epochs=epochs, callbacks = [callback_save,callback_early_stopping])
# stae_model.save("saved_model.h5")
# stae_model.save('saved_model.h5')
#stae_model.save('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\saved_model_new.h5')
stae_model.save('C:\\Users\\Antonio\\Desktop\\Master\\python\\python-surveillance\\saved_model_new.h5py')

