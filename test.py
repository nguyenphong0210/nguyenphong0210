import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet("yolov3_training.cfg",r"yolov3_training_last.weights")

classes = ['pass','er01','er02']

cap = cv2.VideoCapture("3139655933651.mp4")
# cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0
while True:
    frame_id += 1

    _, img = cap.read()
    img = cv2.resize(img,(1280,720))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    font = cv2.FONT_HERSHEY_PLAIN

    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))

            if label == 'pass':
                cv2.rectangle(img,(x,y),(x+w,y+h),(0, 0, 255),1)
                cv2.rectangle(img, (x, y), (x + w, y - 15),(0, 0, 255), -1)
                cv2.putText(img,label + " " + confidence, (x,y-5),font,1,(255, 255, 255),1)
            elif label == 'er01':
                cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 255),2)
                cv2.putText(img,label + " " + confidence, (x,y-5),font,1,(0, 255, 255),2)                
            elif label == 'er02':
                cv2.rectangle(img,(x,y),(x+w,y+h),(255, 255, 255),2)
                cv2.putText(img,label + " " + confidence, (x,y-5),font,1,(255, 255, 255),2)  


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()