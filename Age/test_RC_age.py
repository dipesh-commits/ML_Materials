import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras_vggface.vggface import VGGFace
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.engine import  Model
from keras.layers import Input
import tensorflow as tf
import cv2
import os
from mtcnn.mtcnn import MTCNN

img_width,img_height = 224, 224
input_shape=(img_width, img_height, 3)


model = Sequential()
model.add(VGGFace(model='resnet50', include_top = False, input_shape=(224,224,3), pooling = 'avg'))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8,activation='softmax'))

model.load_weights("/home/user/ML projects/POS/FGNET/resnet Age/weight_resnet.hdf5")
# model.summary()

print(model.inputs)
print(model.outputs)
cap = cv2.VideoCapture(0)
face_detector = MTCNN()

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
val = 10
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
# for images in os.listdir("/home/user/ML projects/POS/face_detection_paper/VGG_face/dataset"):
#     ageList.append(images)
    
print(ageList)


# filename="test_video1.mp4"
# codec=cv2.VideoWriter_fourcc('X','V','I','D')
# frames=20
# res=(640,480)
# videoOutput=cv2.VideoWriter(filename,codec,frames,res)

while True:
    # image = cv2.imread("m.jpg")
    ret, image = cap.read()
    face_people = face_detector.detect_faces(image)
    # print(face_people)
    
    for face in face_people:
        # import pdb;pdb.set_trace()
        try:
            face_boxes = face['box']
            
            face_img = image[face_boxes[1]-val:face_boxes[1]+face_boxes[3]+val,face_boxes[0]-val:face_boxes[0]+face_boxes[2]+val]
            # print(face_img)
            cv2.rectangle(image, (face_boxes[0], face_boxes[1]), (face_boxes[0]+ face_boxes[2],face_boxes[1]+ face_boxes[3]), (0,255,0),2)
            face_img = cv2.resize(face_img, (img_width,img_height))
            face_img = np.expand_dims(face_img, axis=0)
            result = cv2.addWeighted(face_img,2,np.zeros(face_img.shape,face_img.dtype),0,-100)
            agePreds = model.predict(result)
            # agePreds_class = model.predict_classes(face_img)
            # print(agePreds_class)
            age = ageList[agePreds[0].argmax()]
            print(f"{agePreds}: {age}")
            cv2.putText(image, str(age), (face_boxes[0], face_boxes[1]-25),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2)
        except:
            pass

    # videoOutput.write(image)
    # output=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# videoOutput.release()

cv2.destroyAllWindows()
