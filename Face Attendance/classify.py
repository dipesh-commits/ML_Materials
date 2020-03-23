import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from random import choice
import matplotlib.pyplot as plt





data = np.load('face_test_embedded.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(trainX.shape[0],testX.shape[0])


data2 = np.load('face_test.npz')
testX_faces = data2['arr_2']

#preparation for vector normalizing
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)


#one hot encoding
le = LabelEncoder()
le.fit(trainy)
trainy = le.transform(trainy)
testy = le.transform(testy)

#implementing model
model = SVC(kernel='linear',probability =True)
print("Train X:",trainX)
print("Train Y:",trainy)
model.fit(trainX,trainy)

# #predition
# yhat_train = model.predict(trainX)
# yhat_test = model.predict(testX)

# #score
# score_train = accuracy_score(trainy,yhat_train)
# score_test = accuracy_score(testy,yhat_test)

# print("Accuracy: train=%.3f, test=%.3f" %(score_train*100,score_test*100))



selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = le.inverse_transform([random_face_class])

#prediction for the face
samples = np.expand_dims(random_face_emb, axis = 0)
print(samples)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)


class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index]*100
predict_names = le.inverse_transform(yhat_class)

print('Predicted:%s (%.3f)' %(predict_names[0],class_probability))
print('Expected: %s' % random_face_name[0])

plt.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()






