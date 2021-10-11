


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from .utils import to_categorical,ColorizeToPIL
import torchvision
import PIL
from torchvision.transforms import Resize, ToPILImage, ToTensor,\
    RandomHorizontalFlip,RandomVerticalFlip,\
    RandomRotation,RandomAffine





class ImageDataset(Dataset):

    """reading from Pandas file.    """
    def __init__(self, filepaths, Y_torch=None, A_torch=None, W_torch=None, transform=None):
        #pd pandas with
        self.Y_torch = Y_torch
        self.filepaths = filepaths
        self.transform = transform
        self.A_torch = A_torch
        self.W_torch = W_torch


    def __len__(self):
        return len(self.filepaths)

    def update_attributes(self, filepaths=None, A_torch=None, W_torch=None, Y_torch=None):
        if filepaths is not None:
            self.filepaths[:]= filepaths
        if A_torch is not None:
            self.A_torch[:]= A_torch
        if W_torch is not None:
            self.W_torch[:]= W_torch
        if Y_torch is not None:
            self.Y_torch[:]= Y_torch

    def __getitem__(self, idx):
        if isinstance(self.filepaths[idx], str):
            # X = io.imread(self.filepaths[idx])
            X = PIL.Image.open(self.filepaths[idx]).convert("RGB")
            # X = Image.open(self.filepaths[idx]).convert("RGB")
            # print('String')
        elif isinstance(self.filepaths[idx], np.ndarray):
            # print('Array')
            X = self.filepaths[idx]

        Y = self.Y_torch[idx]
        if self.transform:
            X = self.transform(X)
        output = [X, Y]
        if self.A_torch is not None:
            A = self.A_torch[idx]
            output.append(A)
            # return X, Y, A
        if self.W_torch is not None:
            W = self.W_torch[idx]
            output.append(W)
        return output



def get_dataloaders_image(data_pd,file_tag = 'filepath', utility_tag='utility',
                          group_tag = None, weights_tag = None, sampler_tag='weights_sampler',sampler_on=False,
                          augmentations = None, shuffle=True, num_workers = 8,
                          batch_size = 32,regression = False,drop_last=False, group2cat = False,
                          utility2cat = True):

    if not regression:
        n_utility = data_pd[utility_tag].nunique()
        if utility2cat:
            Y_torch = torch.Tensor(to_categorical(data_pd[utility_tag].values, num_classes=n_utility).astype('float32'))
        else:
            Y_torch = torch.Tensor(data_pd[utility_tag].values.astype('float32'))
    else:
        Y_torch  = torch.Tensor(data_pd[utility_tag].values.astype('float32'))

    if group_tag is not None:
        n_group = data_pd[group_tag].nunique()
        if group2cat:
            A_torch = torch.Tensor(to_categorical(data_pd[group_tag].values, num_classes=n_group).astype('float32'))  # same as with utility
        else:
            A_torch = torch.Tensor(data_pd[group_tag].values.astype('float32'))
    else:
        A_torch = None

    if weights_tag is not None:
        W_torch = torch.Tensor(np.vstack(data_pd[weights_tag].values).astype('float32'))
    else:
        W_torch=None

    ## Augmentations
    if augmentations is None:
        augmentations = torchvision.transforms.Compose([ColorizeToPIL(),ToTensor()])

    if sampler_on:
        data_weights = torch.DoubleTensor(data_pd[sampler_tag].values)
        data_sampler = torch.utils.data.sampler.WeightedRandomSampler(data_weights, len(data_weights))
        shuffle = False #shuffle mutually exclusive with balance_sampler
    else:
        data_sampler = None

    filepaths =  data_pd[file_tag].values
    image_dataloader = DataLoader(ImageDataset(filepaths=filepaths, Y_torch=Y_torch, A_torch=A_torch,
                                               W_torch=W_torch, transform=augmentations),batch_size=batch_size,
                                  sampler=data_sampler,shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return image_dataloader
