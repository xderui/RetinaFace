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
from PIL import Image


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
    net = load_model(net, 'weights/PPLCNET0.25_Final_1.pth', True)
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

    # from model import Face_Emotion_CNN
    # from emotion_pplcnet import PPLCNet_x0_5
    # emotion_net = Face_Emotion_CNN()
    # emotion_net.load_state_dict(torch.load('weights/FER_trained_model.pt', map_location=lambda storage, loc: storage), strict=False)
    # emotion_net.eval()
    # emotion_net = emotion_net.cpu()

    from models.minxception import mini_XCEPTION
    from resnet import ResNet101
    # emotion_net = ResNet101(num_classes=10)
    emotion_net = mini_XCEPTION(num_classes=10)
    emotion_net.load_state_dict(torch.load('weights/E223_acc_0.6737.pth', map_location=lambda storage, loc: storage),
                                strict=False)
    emotion_net.eval()
    emotion_net = emotion_net.cpu()



    # testing scale
    resize = 1
    import torchvision.transforms as transforms
    img_transform = transforms.Compose([
        transforms.ToTensor()])

    # emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
    #                 4: 'anger', 5: 'disguest', 6: 'fear'}

    emotion_dict = {0:'neutral',1:'happiness',2:'surprise',3:'sadness',4:'anger',5:'disgust',6:'fear',
                    7:'contempt',8:'unknown',9:'NF'}


    # 视频流
    cv2.namedWindow("Emotion classfication")  # 定义窗口
    cap = cv2.VideoCapture('676685907_1_0.mp4')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # testing begin
    while(True):
        ret, frame = cap.read()
        # frame = cv2.imread('a08b87d6277f9e2f003f414939756822b899f330.jpeg')
        
        # cv2.imwrite('capture.jpg', frame)
        with torch.no_grad():
            # img_raw = cv2.imread(frame)
            img_raw = frame
            # img_raw = cv2.imread('test.jpeg')

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
            if len(dets) ==0:
                continue
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # gray = frame
            for b in dets:
                if b[4] < 0.5:
                    continue
                for i in range(4):
                    if b[i]<0:
                        b[i]=0
                # print(b[1],b[3],b[0],b[2])
                crop_img = cv2.resize(gray[int(b[1]):int(b[3]),int(b[0]):int(b[2])],(48,48),interpolation=cv2.INTER_NEAREST)
                # cv2.imwrite('crop.jpg',crop_img)
                # crop_img = np.float32(crop_img)
                # crop_img = crop_img.transpose(2, 0, 1)

                print(crop_img.shape)
                # crop_img=torch.from_numpy(crop_img).unsqueeze(0)
                cv2.imwrite('croped_img.png',crop_img)
                crop_img = crop_img/256
                crop_img = Image.fromarray((crop_img))
                crop_img = img_transform(crop_img).unsqueeze(0)
                print(crop_img)
                log_ps = emotion_net(crop_img)
                print(log_ps)
                ps = torch.exp(log_ps)
                top_p,top_class = ps.topk(1,dim=1)
                pred = emotion_dict[int(top_class.numpy())]

                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, pred, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # cv2.imwrite('result.png',frame)
            cv2.imshow('Emotion classfication', frame)
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break

