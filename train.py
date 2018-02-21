import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import utils
import os

if not os.path.exists('./model'):
	os.mkdir('model/')

train_loader = utils.load_data_STL10()

is_cuda = torch.cuda.is_available()

cnn = model.get_net()

criterion = nn.CrossEntropyLoss()
if is_cuda:
	criterion = criterion.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

min_loss = 999

print("START TRAINIG!")

for epoch in range(100):
	epoch_loss = 0
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)
		if is_cuda:
			images, labels = images.cuda(), labels.cuda()
		
		optimizer.zero_grad()
		outputs = cnn(images)
		loss = criterion(outputs, labels)
		epoch_loss += loss.data[0]
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
				% (epoch+1, 100, i+1, len(train_loader), loss.data[0]))
	avg_epoch_loss = epoch_loss / len(train_loader)
	print("Epoch: %d, Avg Loss: %.4f" % (epoch+1, avg_epoch_loss))
	if avg_epoch_loss < min_loss:
		print("Renew model")
		min_loss = avg_epoch_loss
		torch.save(cnn.state_dict(), 'model/cnn.pth')
	print("------------------------------")	
