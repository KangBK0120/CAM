import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms

import utils
import model

def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)
    
    test_loader, num_class = utils.get_testloader(config.dataset,
                                        config.dataset_path,
                                        config.img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    cnn.load_state_dict(
        torch.load(os.path.join(config.model_path, config.model_name))
    )
    finalconv_name = 'conv'

    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

    cnn._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(cnn.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (config.img_size, config.img_size)
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    for i, (image_tensor, label) in enumerate(test_loader):
        image_PIL = transforms.ToPILImage()(image_tensor[0])
        image_PIL.save(os.path.join(config.result_path, 'img%d.png' % (i + 1)))

        image_tensor = image_tensor.to(device)
        logit, _ = cnn(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
        img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(config.result_path, 'cam%d.png' % (i + 1)), result)
        if i + 1 == config.num_result:
            break
        feature_blobs.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_result', type=int, default=1)

    config = parser.parse_args()
    print(config)

    create_cam(config)