
import numpy as np
from torchvision.transforms import ToPILImage
import torch
import json



#Keras source code
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def to_np(t):
    return t.cpu().detach().numpy()

class ColorizeToPIL(object):
    """Make Grayscale images into fake RGB images
    """

    def __init__(self):
        self.ToPil = ToPILImage()
        return

    def __call__(self, X):
        if len(X.shape) < 3:
            X = np.repeat(np.array(X)[:, :, np.newaxis], 3, axis=-1)
        X = self.ToPil(X)
        return X

class TravellingMean:
    def __init__(self):
        self.count = 0
        self._mean= 0

    @property
    def mean(self):
        return self._mean

    def update(self, val, mass=None):
        if mass is None:
            mass = val.shape[0]
        if mass > 0:
            self.count+=mass
            self._mean += ((np.mean(val)-self._mean)*mass)/self.count


def model_params_save(filename,classifier_network, optimizer = None):
    if optimizer is not None:
        torch.save([classifier_network.state_dict(),optimizer.state_dict()], filename)
    else:
        torch.save([classifier_network.state_dict()], filename)

def model_params_load(filename,classifier_network, optimizer,DEVICE):

    saves_list = torch.load(filename, map_location=DEVICE)
    classifier_dic = saves_list[0]
    classifier_network.load_state_dict(classifier_dic)

    if optimizer is not None:
        if len(saves_list) > 1:
            optimizer_dic = saves_list[1]
            optimizer.load_state_dict(optimizer_dic)


def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)

def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))
