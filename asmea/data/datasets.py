import os
import asmea.data.transforms as eeea_transforms
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torchvision import datasets
import torchvision.transforms as transforms
from asmea.data.folder2lmdb import ImageFolderLMDB
from asmea.data.autoaugment import ImageNetPolicy
from torchvision import datasets
import torchvision
def get_imagenet_search_queue(args):
	data_dir = os.path.join(args.tmp_data_dir, '')
	traindir = os.path.join(data_dir, 'train')
	validdir = os.path.join(data_dir, 'val')
	normalize = transforms.Normalize(mean=[0.4654, 0.4545, 0.4254], std=[0.2366, 0.2300, 0.2367])
	train_augment = [transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ColorJitter(
					brightness=0.4,
					contrast=0.4,
					saturation=0.4,
					hue=0.2),
					transforms.ToTensor(),
					normalize,]

	val_augment = [transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize,]
	if args.autoaugment: train_augment = transforms.Compose(
		[transforms.Resize(256), ImageNetPolicy(), transforms.ToTensor()])
	if args.lmdb:
		traindir = os.path.join(data_dir, 'train.lmdb')
		validdir = os.path.join(data_dir, 'val.lmdb')

		train_data = ImageFolderLMDB(traindir, transforms.Compose(train_augment))
		valid_data = ImageFolderLMDB(validdir, transforms.Compose(val_augment))
	else:
		train_data = dset.ImageFolder(traindir, transforms.Compose(train_augment))
		valid_data = dset.ImageFolder(validdir, transforms.Compose(val_augment))

	train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=args.workers)

	valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size_val, shuffle=False, pin_memory=True, num_workers=args.workers)
	return train_queue, valid_queue



def get_imagenet_queue(args):
	data_dir = os.path.join(args.tmp_data_dir, '')
	traindir = os.path.join(data_dir, 'train')
	validdir = os.path.join(data_dir, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	train_augment = [transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ColorJitter(
					brightness=0.4,
					contrast=0.4,
					saturation=0.4,
					hue=0.2),
					transforms.ToTensor(),
					normalize,]

	val_augment = [transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize,]

	if args.lmdb:
		traindir = os.path.join(data_dir, 'train.lmdb')
		validdir = os.path.join(data_dir, 'val.lmdb')

		train_data = ImageFolderLMDB(traindir, transforms.Compose(train_augment))
		valid_data = ImageFolderLMDB(validdir, transforms.Compose(val_augment))
	else:

# 		train_data = dset.ImageFolder(traindir, transform= transforms.Compose([transforms.Resize(256),
# 																			   transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(), ImageNetPolicy(), transforms.ToTensor(),transforms.Normalize(std = [0.229, 0.224, 0.225],mean = [0.485, 0.456, 0.406]
# )]))
		train_data = dset.ImageFolder(traindir, transforms.Compose(train_augment))
		valid_data = dset.ImageFolder(validdir, transforms.Compose(val_augment))


	train_queue = torch.utils.data.DataLoader(
		train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

	valid_queue = torch.utils.data.DataLoader(
		valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
	return train_queue, valid_queue

def get_cifar_queue(args):

	if args.dataset == 'cifar10':
		train_transform, valid_transform = eeea_transforms._data_transforms_cifar10(args)
		train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
		valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
	elif args.dataset == 'cifar100':
		train_transform, valid_transform = eeea_transforms._data_transforms_cifar100(args)
		train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
		valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
	elif args.dataset == 'svhn':
		train_transform, valid_transform = eeea_transforms._data_transforms_svhn(args)
		train_data = torchvision.datasets.SVHN(root=args.data, split='train', download=True, transform=train_transform)
		valid_data = torchvision.datasets.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
	elif args.dataset == 'cinic10':
		train_transform, valid_transform = eeea_transforms._data_transforms_cinic10(args)
		train_data = dset.cinic10(root=args.data, train=True, download=True, transform=train_transform)
		valid_data = dset.cinic10(root=args.data, train=False, download=True, transform=valid_transform)
	
	train_queue = torch.utils.data.DataLoader(
	train_data, batch_size=args.batch_size_train, shuffle=True, pin_memory=True, num_workers=args.workers)
	valid_queue = torch.utils.data.DataLoader(
	valid_data, batch_size=args.batch_size_val, shuffle=False, pin_memory=True, num_workers=args.workers)

	return train_queue, valid_queue