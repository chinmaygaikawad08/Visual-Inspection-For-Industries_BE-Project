# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_object(frame, detect, object):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the object detections
	detect.setInput(blob)
	detections = detect.forward()
	print(detections.shape)

	# initialize our list of dimensions, their corresponding locations,
	# and the list of predictions from our object network
	obj = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the objcet ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			item = frame[startY:endY, startX:endX]
			item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
			item = cv2.resize(item, (224, 224))
			item = img_to_array(item)
			item = preprocess_input(item)

			# add the item and bounding boxes to their respective
			# lists
			obj.append(item)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one item was detected
	if len(obj) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		obj = np.array(obj, dtype="float32")
		preds = object.predict(obj, batch_size=32)

	# return a 2-tuple of the item locations and their corresponding
	# locations
	return (locs, preds)
	
# load our serialized object detector model from disk
prototxtPath = r"object_detector\SSD_MobileNet.prototxt"
weightsPath = r"object_detector\SSD_MobileNet.caffemodel"
detect = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the object detector model from disk
object = load_model("object_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect items in the frame and determine if they are perfect or faulty
	(locs, preds) = detect_object(frame, detect, object)

	# loop over the detected object locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(defected, perfect) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "defected" if defected > perfect else "perfect"
		color = (0, 0, 255) if label == "defected" else (0, 255, 0)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(defected, perfect) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()