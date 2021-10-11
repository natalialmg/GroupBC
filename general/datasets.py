
import numpy as np
from .misc import unpickle
import pandas as pd
import os
from sklearn import preprocessing

def CIFAR100(utility_tag = "fine",groups_tag = "coarse", filter_labels = []):
    ## download dataset from : https://www.kaggle.com/fedesoriano/cifar100
    dirdata = '/data/cifar100/'

    metadata_path = dirdata + "meta"
    metadata = unpickle(metadata_path)

    superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
    fineclass_dict = dict(list(enumerate(metadata[b'fine_label_names'])))

    train_path = dirdata + "train"
    test_path = dirdata + "test"

    data_train_dict = unpickle(train_path)
    data_test_dict = unpickle(test_path)

    data_train = data_train_dict[b'data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)  # transpose because we will use toPILImage torchvision and need hxwxc
    data_test = data_test_dict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    totaldata = np.concatenate((data_train, data_test), axis=0)
    finelabels = np.concatenate((data_train_dict[b'fine_labels'], data_test_dict[b'fine_labels']), 0)
    coarselabels = np.concatenate((data_train_dict[b'coarse_labels'], data_test_dict[b'coarse_labels']), 0)
    filenames = np.concatenate((data_train_dict[b'filenames'], data_test_dict[b'filenames']), 0)
    dataset = ['train' for i in range(data_train.shape[0])]
    dataset.extend(['test' for i in range(data_test.shape[0])])
    coarselabels_names = [superclass_dict[coarse_ix] for coarse_ix in coarselabels]
    finelabels_names = [fineclass_dict[class_ix] for class_ix in finelabels]

    totaldata = [totaldata[i] for i in
                 range(totaldata.shape[0])]  # no need to divide by 255 because we will use toPILImage torchvision

    pd_data = pd.DataFrame()
    pd_data["data"] = totaldata
    pd_data["sample_index"] = [i for i in range(len(totaldata))]
    pd_data["coarse"] = coarselabels.tolist()
    pd_data["fine"] = finelabels.tolist()
    pd_data["filename"] = filenames.tolist()
    pd_data["dataset"] = dataset
    pd_data["coarse_tags"] = coarselabels_names
    pd_data["fine_tags"] = finelabels_names

    ## utility are fine tags, groups are coarse tags
    pd_data["utility"] = pd_data[utility_tag]
    pd_data["utility_str"] = pd_data[utility_tag+ "_tags"]
    pd_data["groups"] = pd_data[groups_tag]
    pd_data["groups_str"] = pd_data[groups_tag + "_tags"]

    if len(filter_labels)>1:
        new_label = {}
        ix = 0
        for label in filter_labels:
            new_label[label] = ix
            ix += 1

        filter = [True if label in filter_labels else False for label in pd_data['utility'].values]
        pd_data = pd_data.loc[filter]

        #rename label
        pd_data['utility'] = [new_label[u] for u in pd_data['utility'].values]



    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']



    return pd_train,pd_test


def CelebA(utility="Blond_Hair", group_tag = 'Male'): #, split=1, n_splits=5, seed=42):
    dirdata = '/home/natalia/celeba-dataset/'
    pd_data = pd.read_csv(os.path.join(dirdata, 'processed_celeba.csv'), index_col=0)
    file_list = []
    for file in pd_data['image_id'].values:
        file_list.append(dirdata + 'img_align_celeba/img_align_celeba/' + file)
    # print(file_img[-1])
    pd_data['filepath'] = file_list

    pd_data = pd_data.reset_index()
    pd_data['sample_index'] = np.arange(len(pd_data))
    values = pd_data[utility].values
    values[values == -1] = 0
    pd_data["utility"] = values
    pd_data['group_tag'] = pd_data[utility].astype(str) + pd_data[group_tag].astype(str)
    le = preprocessing.LabelEncoder()
    group_int = le.fit_transform(pd_data['group_tag'])
    pd_data['group'] = group_int

    pd_test = pd_data.loc[pd_data['partition'] == 2]
    pd_train = pd_data.loc[pd_data['partition'] != 2]

    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=n_splits, random_state=seed)
    # skf.get_n_splits(pd_data)
    # index_splits = skf.split(pd_data, pd_data["utility"].values)
    #
    # ix = 0
    # for train_index, test_index in index_splits:
    #     pd_train, pd_test = pd_data.iloc[train_index], pd_data.iloc[test_index]
    #     if ix == split:
    #         print("split:", ix)
    #         break
    return pd_train, pd_test


def CUB(utility="y"):
    dirdata = '/home/natalia/waterbird-dataset/waterbird_complete95_forest2water2/'
    pd_data = pd.read_csv(os.path.join(dirdata, 'metadata.csv'), index_col=0)
    file_list = []
    for file in pd_data['img_filename'].values:
        file_list.append(dirdata + file)
    # print(file_list[-1])
    pd_data['filepath'] = file_list
    pd_data['sample_index'] = np.arange(len(pd_data))
    pd_data['utility'] = pd_data[utility].values

    #groups are a combination of Y (utility) and place
    pd_data['group_tag'] = pd_data['y'].astype(str) + pd_data['place'].astype(str)

    le = preprocessing.LabelEncoder()
    group_int = le.fit_transform(pd_data['group_tag'])
    pd_data['group'] = group_int

    pd_train = pd_data.loc[pd_data['split'] != 2]
    pd_test = pd_data.loc[pd_data['split'] == 2]


    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=n_splits, random_state=seed)
    # skf.get_n_splits(pd_data)
    # index_splits = skf.split(pd_data, pd_data["utility"].values)
    #
    # ix = 0
    # for train_index, test_index in index_splits:
    #     pd_train, pd_test = pd_data.iloc[train_index], pd_data.iloc[test_index]
    #     if ix == split:
    #         print("split:", ix)
    #         break

    return pd_train, pd_test


def CIFAR10():

    dirdata = '/data/cifar-10-python/'
    ###  train
    dataset = []
    for ibatch in range(5):
        metadata_path = dirdata + "data_batch_" + str(ibatch + 1)
        metadata = unpickle(metadata_path)

        data_train = metadata[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        if ibatch == 0:
            totaldata = np.array(data_train)
            totallabels = np.array(metadata[b'labels'])
            totalnames = np.array(metadata[b'filenames'])
        else:
            totaldata = np.concatenate((totaldata, data_train), axis=0)
            totallabels = np.concatenate((totallabels, metadata[b'labels']), 0)
            totalnames = np.concatenate((totalnames, metadata[b'filenames']), 0)
        dataset.extend(['train' for i in range(data_train.shape[0])])

    ### test
    metadata_path = dirdata + "test_batch"
    metadata = unpickle(metadata_path)
    data_train = metadata[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    totaldata = np.concatenate((totaldata, data_train), axis=0)
    totallabels = np.concatenate((totallabels, metadata[b'labels']), 0)
    totalnames = np.concatenate((totalnames, metadata[b'filenames']), 0)
    dataset.extend(['test' for i in range(data_train.shape[0])])

    ### consolidate pandas
    labels_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    totaldata = [totaldata[i] for i in
                 range(totaldata.shape[0])]  # no need to divide by 255 because we will use toPILImage torchvision
    pd_data = pd.DataFrame()
    pd_data["data"] = totaldata
    pd_data["sample_index"] = [i for i in range(len(totaldata))]
    pd_data["labels"] = totallabels.tolist()
    pd_data["filename"] = totalnames.tolist()
    pd_data["dataset"] = dataset

    ## utility are fine tags, groups are coarse tags
    pd_data["utility"] = pd_data["labels"].values
    pd_data["utility_str"] = [labels_name[i] for i in pd_data["labels"].values]

    pd_train = pd_data.loc[pd_data.dataset == 'train']
    pd_test = pd_data.loc[pd_data.dataset == 'test']

    return pd_train, pd_test