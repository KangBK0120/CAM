import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import model
import utils

def train(config):
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    train_loader, num_class = utils.get_trainloader(config.dataset, 
                                        config.dataset_path,
                                        config.img_size,
                                        config.batch_size)
                                        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.lr)

    min_loss = 999

    print("START TRAINING")
    for epoch in range(config.epoch):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % config.log_step == 0:
                if config.save_model_in_epoch:
                    torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, i + 1, len(train_loader), loss.item()))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Loss: %.4f'
                    % (epoch + 1, config.epoch, avg_epoch_loss))
        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            torch.save(cnn.state_dict(), os.path.join(config.model_path, config.model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-s', '--save_model_in_epoch', action='store_true')
    config = parser.parse_args()
    print(config)

    train(config)