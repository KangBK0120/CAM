import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

transform = transforms.Compose([
	transforms.Resize(128),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform2 = transforms.Compose([
	transforms.Resize(128),
	transforms.ToTensor()
])
def load_data_STL10(batch_size=64, test=False):
	if not(test):
		train_dset = dsets.STL10(root='./data', split='train', transform=transform2, download=True)
	else:
		train_dset = dsets.STL10(root='./data', split='test', transform=transform2, download=True)
	train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	print("LOAD DATA, %d"%(len(train_loader)))
	return train_loader

def load_data_CIFAR(batch_size=64):
	train_dset = dsets.CIFAR10(root='./data', train=True, transform=transform, download=True)
	train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	print("LOAD DATA, %d" % (len(train_loader)))
	return train_loader
