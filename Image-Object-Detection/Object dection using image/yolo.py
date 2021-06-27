# USAGE
# python yolo.py --image images/sam.jpg


import numpy as np
import argparse
import time
import cv2
import os

# construire l'argument parse et parser les arguments.

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# Charger les étiquettes de classe COCO sur lesquelles le modele YOLO a été entrainé

labelsPath = 'coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialiser une liste de couleurs pour représenter chaque étiquette de classe possible
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# dériver les chemins vers les weights YOLO et la configuration du modèle
weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'

# charger notre détecteur d'objets YOLO formé sur le jeu de données COCO (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# charger notre image d'entrée et saisir ses dimensions spatiales
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# déterminer uniquement les noms de couche *output* dont nous avons besoin de YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construire un blob à partir de l'image d'entrée, puis effectuer un transfert
# passage du détecteur d'objet YOLO, nous donnant nos cadres de délimitation et probabilités associées
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()


print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialiser nos listes de cadres de délimitation détectés, de confidences et ID de classe, respectivement
boxes = []
confidences = []
classIDs = []


for output in layerOutputs:
	for detection in output:
		
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filtrer les prédictions faibles en assurant la détection
		# la probabilité est supérieure à la probabilité minimale
		if confidence > args["confidence"]:
    			
			# redimensionner les coordonnées du cadre de délimitation par rapport a la taille de l'image, 
			# YOLO renvoie les coordonnées centrales (x, y) de délimitation
			
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])


if len(idxs) > 0:
	
	for i in idxs.flatten():
		
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# afficher l'image de sortie
cv2.imshow("Image", image)
cv2.waitKey(0)