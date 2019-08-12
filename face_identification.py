import os
import pickle
import cv2 as cv
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv.CascadeClassifier('cascades/data'
	+ '/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

x_trains = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
	for file in files:

		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).lower()
			
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]

			pil_image = Image.open(path).convert("L")
			size = (500,500)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")

			faces = face_cascade.detectMultiScale(image_array, 
				scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_trains.append(roi)
				y_labels.append(id_)

with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)

recognizer.train(x_trains, np.array(y_labels))
recognizer.save("trainer.yml")