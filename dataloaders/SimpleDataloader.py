from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.ImageDataset import ImageDataset
import torch


def basicImageDataloader(train_path_list, test_path_list, num_iters=None, crop_size=128, train_batch=1, test_batch=1,**kwargs):

    batch_sizes = {'train': train_batch, 'test':test_batch}

    train_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}

    image_datasets = {'train': ImageDataset(train_path_list, data_transforms['train'], num_iters=num_iters),
                      'test': ImageDataset(test_path_list, data_transforms['test'])}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_sizes[x], shuffle=(x == 'train'),**kwargs) for x in ['train', 'test']}

    return dataloaders


