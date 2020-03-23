import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from mtcnn.mtcnn import MTCNN




train_dir = 'data/train'

valid_dir = 'data/val'

face_detector = MTCNN()

# for i in os.listdir(train_dir):
# 	print(i)

# my_img = 'data/train/madonna/httpiamediaimdbcomimagesMMVBMTANDQNTAxNDVeQTJeQWpwZBbWUMDIMjQOTYVUXCRALjpg.jpg'
# img = img.convert("RGB")


def extract_faces(filename):
	img_path = filename
	img = Image.open(img_path)
	img = img.convert("RGB")
	pixels = np.asarray(img)
	results = face_detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1 = abs(x1)
	y1 = abs(y1)
	x2,y2 = x1+width, y1+height
	face = pixels[y1:y2,x1:x2]

	image = Image.fromarray(face)
	resized_img = image.resize((160,160))
	final_pix = np.asarray(resized_img)



	return final_pix



def load_faces(directory):
	faces = []
	for filename in os.listdir(directory):
		path = os.path.join(directory,filename)
		face = extract_faces(path)
		faces.append(face)
	return faces


def load_dataset(directory):
	X, y = [], []
	for subdir in os.listdir(directory):

		path = directory +'/' + subdir + '/'

		if not os.path.isdir(path):
			continue
		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]

		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)


# load_dataset(train_dir))

trainX, trainy = load_dataset(train_dir)
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(valid_dir)
print(testX.shape, testy.shape)
# save arrays to one file in compressed format
# np.savez_compressed('face_test.npz', trainX, trainy, testX, testy)



# plt.imshow(ans)
# plt.show()