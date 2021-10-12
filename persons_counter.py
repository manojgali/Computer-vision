import cv2
import numpy as np
#from scipy.spatial.distance import cdist

net = cv2.dnn.readNet('models/yolov4.weights','models/yolov4.cfg')

classes=[]
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()
try:
    cap = cv2.VideoCapture('dipr/10.mp4')
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (416,416),(0,0,0),swapRB = True, crop = False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        frame = 1
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4 )
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        count = 0
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                count= count + 1
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label+ " "+ confidence, (x,y+20), font, 2, (255,255,255), 2, )
        cv2.putText(img,"number of people per frame :" + str(count), (10,30), font, 2, (0,255,255), 2,)                       
        cv2.imshow('frame', img)
        key = cv2.waitKey(10)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except AttributeError:
    print("video ended ")
except ValueError:
    print("video ended ")
cap.release()
cv2.destroyAllWindows()