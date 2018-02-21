import torch
import torch.nn

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv = nn.Sequential(
			# 3 x 128 x 128
			nn.Conv2d(3, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),

			# 32 x 128 x 128
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			# 64 x 128 x 128
			nn.MaxPool2d(2, 2),
			
			# 64 x 64 x 64
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			
			# 128 x 64 x 64
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			
			# 256 x 64 x 64
			nn.MaxPool2d(2, 2)
			
			# 256 x 32 x 32
			nn.Conv2d(256, 10, 3, 1, 1),
			nn.BatchNorm2d(10),
			nn.LeakyReLU(0.2)
		)
		# 256 x 32 x 32
		self.avg_pool = nn.AvgPool2d(32)
		# 256 x 1 x 1
		self.classifier = nn.Linear(10, 10)

	def forward(self, x):
		features = self.conv(x),
		flatten = self.avg_pool(features).view(features.size(0), -1)
		output = self.classifier(flatten)
		return output, features

def weight_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def get_net():
	net = CNN()
	net.apply(weights_init)
	if torch.cuda.is_available():
		print("USING CUDA")
		net.cuda()
	print("INIT NETWORK")
	return net

def load_net():
	net = CNN()
	net.load_state_dict(torch.load("model/cnn.pth"))
	if torch.cuda.is_available():
		net.cuda()
	return net
