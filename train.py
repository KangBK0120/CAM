import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import utils

if not os.path.exists('./model'):
    os.mkdir('model/')

train_loader = utils.load_data_stl10()

is_cuda = torch.cuda.is_available()

device = torch.device("cuda" if is_cuda else "cpu")
cnn = model.get_net().to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

min_loss = 999

print("START TRAINING")

for epoch in range(100):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = cnn(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                  % (epoch+1, 10, i+1, len(train_loader), loss.item()))

    avg_epoch_loss = epoch_loss / len(train_loader)
    print("Epoch: %d, Avg Loss: %.4f" % (epoch+1, avg_epoch_loss))
    if avg_epoch_loss < min_loss:
        print("Renew model")
        min_loss = avg_epoch_loss
        torch.save(cnn.state_dict(), 'model/cnn.pth')
    print("----------------------------------")
