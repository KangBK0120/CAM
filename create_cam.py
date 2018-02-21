import model
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
import utils
import os
import torchvision.transforms as transforms
from torch.nn import functional as F

if os.path.exists('./result'):
	os.mkdir('result/')

test_loader = utils.load_data_STL10(batch_size=1, test=True)

net = model.load_net()
finalconv_name ='conv'

# hook
feature_blobs = []
def hook_feature(module, input, output):
	feature_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
# get only weight from last layer(linear)
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
	size_upsample = (128, 128)
	bz, nc, h, w = feature_conv.shape
	output_cam = []
	for idx in class_idx:
		cam = weight_softmax[class_idx].dot(feature_conv.reshape( (nc, h*w)))
		cam = cam.reshape(h, w)
		cam = cam - np.min(cam)
		cam_img = cam/np.max(cam)
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))
	return output_cam

image_tensor, _ = next(iter(test_loader))
image_var = Variable(image_tensor)
image_PIL = transforms.ToPILImage()(image_tensor[0])
image_PIL.save('test.jpg')

if torch.cuda.is_available():
	image_var = image_var.cuda()
print("CAL LOGIT")
logit = net(image_var)
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
print(idx[0])
CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0]])

img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)

