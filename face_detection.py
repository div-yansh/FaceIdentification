import pickle
import cv2 as cv

face_cascade = cv.CascadeClassifier('cascades/data'
	+ '/haarcascade_frontalface_alt2.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, 
		minNeighbors=5)

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]

		id_, confidence = recognizer.predict(roi_gray)

		if confidence >= 85:
			name = labels[id_]
			font = cv.FONT_HERSHEY_SIMPLEX
			color = (0,255,0)
			stroke = 1
			cv.putText(frame, name, (x,y), font, 1, color, cv.LINE_AA)

		color = (0,255,0)
		stroke = 2
		cv.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

	cv.imshow("Frame", frame)
	if cv.waitKey(1) & 0XFF == 27:
		break

cap.release()
cv.destroyAllWindows()