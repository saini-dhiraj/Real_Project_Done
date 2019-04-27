###############################               Predicting Tennis ball          ######################################################

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json
from google.colab.patches import cv2_imshow
from tensorflow.keras.optimizers import Adam

with open('//content//drive//My Drive//test//model_in_json.json','r') as f:
  model_json = json.load(f)

model = model_from_json(model_json)

model.load_weights('//content//drive//My Drive//weights_tennis.h5','r')
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

x=32
h= 40

video = cv2.VideoCapture("//content//drive//My Drive//test//B.mp4")
print("loading Video complete")

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (299,299))

while (video.isOpened()):
  ret,frame = video.read()
  #Convert the captured frame into RGB
  
  if ret == True:
    im = Image.fromarray(frame, 'RGB')
    #Resizing into 299x299 because we trained the model with this image size.
    im = im.resize((299,299))
    img_array = np.array(im)
  
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x299x299x3 
    
    img = np.expand_dims(img_array, axis=0)

    #Calling the predict method on model to predict 'tennis' in the image
  
    prediction = int(model.predict(img)[0][1])

    #if prediction is 1, which means I am detecting the image, then show the frame in gray color.
  
    if prediction == 1:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      text = "Tennis Ball"
      cv2.putText(img_array, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
      out.write(img_array)
    else:
      text = "Unknown"
      cv2.putText(img_array, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
      out.write(img_array)
    
    key=cv2.waitKey(1)
    if key == ord('q'):
      break
print("Complete Execution ")
out.release()    
video.release()
cv2.destroyAllWindows()