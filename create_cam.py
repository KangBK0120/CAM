import os
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
import utils
import model

if not os.path.exists('./result'):
    os.mkdir('result/')

classes = ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey',
        'ship', 'truck')
test_loader = utils.load_data_stl10(batch_size=1, test=True)
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
net = model.load_net().to(device)
finalconv_name = 'conv'

# hook
feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
# get weight only from the last layer(linear)
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

image_tensor, _ = next(iter(test_loader))
image_PIL = transforms.ToPILImage()(image_tensor[0])
image_PIL.save('result/test.jpg')

image_tensor = image_tensor.to(device)
logit, _ = net(image_tensor)
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
print(idx[0].item(), classes[idx[0]], probs[0].item())
CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
img = cv2.imread('result/test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('result/CAM.jpg', result)
