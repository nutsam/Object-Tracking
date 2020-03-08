import time
import numpy as np
import cv2
from cv2 import dnn

#make label and color list
labels = ["background", "bicycle", "bus", "car", "motorbike", "person"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

#prepare test image
image = cv2.imread('test_image.jpg')
(h, w) = image.shape[:2]

 #pre-processing
inputScale = 1.0/127.5
inputSize = (512, 512)
inputMean = (127.5, 127.5, 127.5)
blob = dnn.blobFromImage(image, inputScale, inputSize, inputMean, swapRB=False)

#prepare model network
prototxt = r"Mobilenet-SSD_deploy.prototxt"
model = r"Mobilenet-SSD.caffemodel"
net = dnn.readNetFromCaffe(prototxt, model)

#feed in image and get result
net.setInput(blob)
t = time.time()
prob = net.forward()
print("Runtime: %.3f sec" %(time.time()-t))

#diaplay result
pic = image
for i in np.arange(0, prob.shape[2]):
    confidence = prob[0, 0, i, 2]
    if confidence > 0.2:    #change threshold value to get the result you want 
        # get data from prob
        index = int(prob[0, 0, i, 1])
        box = prob[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, endX, endY) = box.astype("int")
        color = colors[index]
        # draw rect
        cv2.rectangle(pic, (x, y), (endX, endY), color, 2)
        # draw label
        label = "{}: {:.2f}".format(labels[index], confidence)
        # print("{}".format(label))
        (fontX, fontY) = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        y = y + fontY if y-fontY<0 else y
        cv2.rectangle(pic,(x, y-fontY),(x+fontX, y),color,cv2.FILLED)
        cv2.putText(pic, label, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)


cv2.imshow("Image", pic)
cv2.waitKey(0)