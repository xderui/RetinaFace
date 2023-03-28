import cv2

cv2.namedWindow("MoblieNet")  # 定义窗口
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

from models.net import PPLCNet_x0_25,ShuffleNet,ghost_net,MobileNetV1,MobileNetV3


features = {
    "0.5x": [24, 48, 96, 192, 1024],
    "1x": [24, 116, 232, 464, 1024],
    "1.5x": [24, 176, 352, 704, 1024],
    "2x": [24, 244, 488, 976, 2048]
}

net = PPLCNet_x0_25()
# net = ShuffleNet(features["0.5x"], [3, 7, 3])
# net = MobileNetV3()
# net = MobileNetV1()


import torch
img = torch.randn(1,3,224,224)
import time
while True:
    ret, frame = cap.read()
    with torch.no_grad():
        s = time.time()
        net(img)
        e = time.time()
    cv2.putText(frame,str(e-s),(0,30),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow('PPLCNet', frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break
