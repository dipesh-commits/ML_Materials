import numpy as np
import cv2
from keras.models import load_model


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
model = load_model('model/model_checkpoint.h5')

def predict_gesture(gesture):
    img = cv2.resize(gesture,(50,50))
    img = img.reshape(1,50,50,1)
    img = img/255
    pred = np.argmax(model.predict(img))
    return classes[pred]


cap = cv2.VideoCapture(-1)
ret, frame = cap.read()

old_text = ''
predicted_text = ''
count_frames = 0
total_str = ''
flag = False

while True:

    if frame is not None:

        frame = cv2.resize(frame,(600,600))
        cv2.rectangle(frame,(200,200),(400,400),(0,255,0),4)
        crop_image = frame[200:400,200:400]
        gray = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
        adaptive_thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
        blackboard = np.zeros(frame.shape, dtype=np.uint8)

        cv2.putText(blackboard, "Predicted text - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))

        if count_frames > 20 and predicted_text == "":
            total_str += predicted_text
            count_frames = 0

        if flag == True:
            old_text = predicted_text
            predicted_text = predict(adaptive_thres)

            if old_text == predicted_text:
                count_frames += 1
            else:
                count_frames += 0

            cv2.putText(blackboard, total_str, (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))



        res = np.hstack((frame, blackboard))

        cv2.imshow('Prediction',res)
        cv2.imshow('hand', adaptive_thres)




    ret, frame = cap.read()

    keypress = cv2.waitKey(1)

    if keypress == ord('c'):
        flag = True
    if keypress == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

        



