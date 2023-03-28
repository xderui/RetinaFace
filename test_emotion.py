from __future__ import print_function
import os
import argparse
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


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
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

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, 'weights/PPLCNET0.5_Final.pth', True)
    net.eval()

    # input_names = ["input"]
    # output_names = ["out"]
    # data = torch.rand(1, 3, 480, 640)
    # torch.onnx._export(net, data, 'mobilenet.onnx', export_params=True, opset_version=11, input_names=input_names,
    #                    output_names=output_names, dynamic_axes={'input': [2, 3], 'out': [2, 3]})
    #
    # net = onnxruntime.InferenceSession("mobilenet.onnx")

    cudnn.benchmark = True
    device = torch.device("cpu")
    net = net.to(device)
    #
    # net = onnxruntime.InferenceSession('mobilenet.onnx')



    # testing scale
    resize = 1




    # 视频流
    cv2.namedWindow("MoblieNet")  # 定义窗口
    cap = cv2.VideoCapture('video.mp4')

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # testing begin
    while(True):
        ret, frame = cap.read()
        # cv2.imwrite('capture.jpg', frame)
        with torch.no_grad():
            # img_raw = cv2.imread(frame)
            # img_raw = frame
            img_raw = cv2.imread('Screenshot_20220411_112218.jpg')
            # img = cv2.imread('test_img.jpg')
            img = np.float32(img_raw)
            # print(img[0][0])
            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= rgb_mean
            # print(img[0][0])

            # print(img.shape)
            img = img.transpose(2, 0, 1)
            # print(img[0][0])
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)
            # print(img)
            # print(img.shape)
            s1 = time.time()
            loc, conf = net(img)  # forward pass
            # print(loc)
            s2 = time.time()
            print(s2-s1)
            # loc, conf, _ = net.run([], {"input": img.cpu().numpy()})
            loc = torch.Tensor(loc)
            conf = torch.Tensor(conf)
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            # print(boxes)
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > 0.2 )[0]
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

            for b in dets:
                if b[4] < 0.4:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imshow('MoblieNet', frame)
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break

