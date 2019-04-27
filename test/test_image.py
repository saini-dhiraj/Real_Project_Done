import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json


with open('model_in_json.json','r') as f:
  model_json = json.load(f)

model = model_from_json(model_json)

model.load_weights('//content//drive//My Drive//weights_tennis.h5','r')
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


img = cv2.imread('//content//drive//My Drive//test//egg.jpg')
#img = cv2.imread('//content//drive//My Drive//test//test.jpg')
img = Image.fromarray(img, 'RGB')
img = img.resize((299,299))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

c = int (model.predict(img_array)[0][1])

if(c==1):
  print("Tennis Ball")

else:
  print ("Unknown")
