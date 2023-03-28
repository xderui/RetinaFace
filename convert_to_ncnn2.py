import torch
from models.retinaface import RetinaFace
from data.config import cfg_mnet


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

cfg = cfg_mnet

# net = RetinaFace(cfg=cfg, phase='test')
# # net = load_model(net, 'weights/PPLCNET0.5_Final.pth', True)
# net.eval()
# device = torch.device("cpu")
# net = net.to(device)
# x = torch.randn(1, 3, 640, 640).to(device)
# torch.onnx._export(net, x, 'retinaface.onnx', opset_version=11,export_params=True)

from models.minxception import mini_XCEPTION
emotion_net = mini_XCEPTION(num_classes=10)
emotion_net.load_state_dict(torch.load('weights/xminception_final1.pth', map_location=lambda storage, loc: storage),
                            strict=False)
emotion_net.eval()
device = torch.device('cpu')
emotion_net = emotion_net.to(device)
x = torch.randn(1,1,48,48)
torch.onnx._export(emotion_net, x, 'minxcpetion.onnx', opset_version=11,export_params=True)