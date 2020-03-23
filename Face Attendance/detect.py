import numpy as np
from keras.models import load_model

data = np.load('face_test.npz')
# print(data['arr_0'])
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


model = load_model('facenet_keras.h5')
print('Model Loaded')


def get_embeddings(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels-mean)/std
	samples = np.expand_dims(face_pixels,axis=0)
	yhat = model.predict(samples)
	return yhat[0]


newTrainX = []
for face_pixels in trainX:
	embedding = get_embeddings(model,face_pixels)
	newTrainX.append(embedding)

newTrainX = np.asarray(newTrainX)

newTestX = []
for face_pixels in testX:
	embedding = get_embeddings(model,face_pixels)
	newTestX.append(embedding)

newTestX = np.asarray(newTestX)

print(newTrainX.shape)
print(newTestX.shape)

np.savez_compressed('face_test_embedded.npz', newTrainX,trainy,newTestX,testy)




