import torch
import torchvision.models as models
from models.minxception import mini_XCEPTION

net = mini_XCEPTION(num_classes=10)
net.load_state_dict(torch.load('weights/E18_acc_0.7347.pth', map_location=lambda storage, loc: storage),
                                strict=False)
net = net.eval()
net = net.cpu()

x = torch.rand(1, 1, 48, 48)

mod = torch.jit.trace(net, x)
torch.jit.save(mod, "minixception.pt")