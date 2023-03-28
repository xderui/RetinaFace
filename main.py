#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout,QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer,Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
import math

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import time
from PIL import Image


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, 'Failed'
    return True


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

rgb_mean = (104, 117, 123)
emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear',
                        7: 'contempt', 8: 'unknown', 9: 'NF'}
reverse_emotion_dict={'neutral':0,'happiness':1,'surprise':2,'sadness':3, 'anger':4, 'disgust':5, 'fear':6,
                        'contempt':7, 'unknown':8, 'NF':9}

class win(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('智能情绪快速识别系统')
        self.img = None   # store
        self.class_num = np.zeros((1,10))
        self.colors=['#F0F8FF','#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF', '#F5F5DC','#FFE4C4','#000000', '#FFEBCD', '#0000FF']
        seaborn_color=['marigold','grapefruit','yellowgreen','cool green','dark sky blue','violet pink','muted blue','chartreuse','coral pink','orangey yellow']
        self.colors2=[]
        for i in seaborn_color:
            self.colors2.append(sns.xkcd_rgb[i])
        self.initUI()
        self.initNet()
        self.timer=QTimer()
        self.timer.timeout.connect(self.time_schedule)
        self.draw_timer=QTimer()
        self.draw_timer.timeout.connect(self.draw_schedule)
        self.start()

    def initNet(self):
        torch.set_grad_enabled(False)
        self.cfg = cfg_mnet
        # net and model
        self.retinaface = RetinaFace(cfg=self.cfg, phase='test')
        print(os.getcwd())
        self.retinaface = load_model(self.retinaface, 'weights/PPLCNET0.25_Final_1.pth', True)  # pre: PPLCNET0.25_Final_1.pth
        self.retinaface.eval()
        cudnn.benchmark = True
        device = torch.device("cpu")
        self.retinaface = self.retinaface.to(device)

        # minixception
        from models.minxception import mini_XCEPTION
        from resnet import ResNet101
        # emotion_net = ResNet101(num_classes=10)
        self.minix = mini_XCEPTION(num_classes=10)
        self.minix.load_state_dict(
            torch.load('weights/E370_acc_0.6867.pth', map_location=lambda storage, loc: storage),
            strict=False)
        self.minix.eval().cpu()

        # process
        resize = 1
        import torchvision.transforms as transforms
        self.img_transform = transforms.Compose([
            transforms.ToTensor()])


    def initUI(self):
        self.resize(1280, 580)
        self.label1 = QLabel()
        self.label2 = QLabel()
        self.label3 = QLabel()
        self.label1.setAlignment(Qt.AlignTop)
        self.label2.setAlignment(Qt.AlignTop)
        self.label3.setAlignment(Qt.AlignVCenter)

        # 布局设定
        layout = QGridLayout(self)
        layout.addWidget(self.label1,0,0,3,7)
        layout.addWidget(self.label2,0,7,2,4)
        layout.addWidget(self.label3,2,8,1,1)
        # 信号与槽连接, PyQt5与Qt5相同, 信号可绑定普通成员函数
        # self.btnOpen.clicked.connect(self.start)

    def start(self):
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.timer.start(40)
        self.draw_timer.start(1200)

    def time_schedule(self):
        if self.cap:
            ret, frame = self.cap.read()
            # img = cv2.resize(frame,(400,280))
            # img = cv2.filp(img,1)
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # print(img.shape)
            img = cv2.resize(img,(int(img.shape[1]),int(img.shape[0])))
            self.img = img


            # start check

            img = np.float32(img)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= rgb_mean
            img = img.transpose(2, 0, 1)
            # print(img[0][0])
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to('cpu')
            scale = scale.to('cpu')
            # print(img)
            # print(img.shape)
            s1 = time.time()
            loc, conf = self.retinaface(img)  # forward pass
            # print(loc)
            s2 = time.time()
            print(s2 - s1)
            # loc, conf, _ = net.run([], {"input": img.cpu().numpy()})
            loc = torch.Tensor(loc)
            conf = torch.Tensor(conf)
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to('cpu')
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale
            # print(boxes)
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > 0.2)[0]
            boxes = boxes[inds]
            # landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            # order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]

            scores = scores[order]
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.3)

            dets = dets[keep, :]
            # print(dets)
            if len(dets) != 0:
                self.class_num = np.zeros((1, 10)).tolist()[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = frame
                for b in dets:
                    if b[4] < 0.5:
                        continue
                    for i in range(4):
                        if b[i] < 0:
                            b[i] = 0
                    # print(b[1],b[3],b[0],b[2])
                    crop_img = cv2.resize(gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])], (48, 48), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite('crop.jpg', crop_img)
                    crop_img = np.float32(crop_img)
                    # crop_img = crop_img.transpose(2, 0, 1)

                    print(crop_img.shape)
                    # crop_img=torch.from_numpy(crop_img).unsqueeze(0)
                    cv2.imwrite('croped_img.png', crop_img)
                    crop_img = crop_img / 256
                    crop_img = Image.fromarray((crop_img))
                    crop_img = self.img_transform(crop_img).unsqueeze(0)

                    log_ps = self.minix(crop_img)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    pred = emotion_dict[int(top_class.numpy())]
                    self.class_num[int(top_class.numpy())] += 1
                    b = list(map(int, b))
                    cv2.rectangle(self.img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(self.img, pred, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            showimg = QImage(self.img.data, self.img.shape[1], self.img.shape[0], QImage.Format_RGB888)
            self.label1.setPixmap(QPixmap.fromImage(showimg))

    def draw_schedule(self):
        # bar
        class_dict = {}
        for i,v in enumerate(self.class_num):
            class_dict[emotion_dict[i]]=v
        plt.figure(figsize=(6,4))
        class_dict=dict(sorted(class_dict.items(),key=lambda x:x[1]))
        new_color=[]
        for k,v in class_dict.items():
            new_color.append(self.colors2[reverse_emotion_dict[k]])
        plt.barh(np.arange(10),class_dict.values(),color=new_color)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.yticks(np.arange(10),list(class_dict.keys()))
        plt.savefig('class_bar.png',dpi=80,transparent=True)
        # plt.margins(10,10)
        plt.tight_layout()
        self.label2.setPixmap(QtGui.QPixmap('class_bar.png'))
        show_label=[]
        color_list=[]
        for k,v in class_dict.items():
            if v!=0:
                show_label.append(k)
                color_list.append(self.colors2[reverse_emotion_dict[k]])
        # pie
        plt.figure(figsize=(3,5))
        plt.pie(class_dict.values(),colors=color_list)
        # plt.legend(list(class_dict.keys()),bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

        plt.legend(show_label,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.0, 1.3),
            borderaxespad=0,
            fontsize=10)
        plt.savefig('class_pie.png',dpi=80,transparent=True)
        # plt.tight_layout()
        self.label3.setPixmap(QtGui.QPixmap('class_pie.png'))

if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())